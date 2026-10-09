"""Private model registry; bounded GGUF metadata and explicit pinned downloads."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import Request, urlopen
from .contracts import AppError, InferenceSettings, ModelSpec, inspect_reasoning_template

SHARD = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', re.I)
DRAFT = re.compile(r'(?:^|[._/-])(?:mtp|draft|eagle3?|dflash|dspark)(?:[._/-]|$)', re.I)
QUANT = re.compile(r'(?:^|[._-])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|F16|BF16|F32)(?:[._-]|$)', re.I)


def _remote_head(item: dict, listing: list[dict]) -> bool:
    """Recognize paired MTP main variants; GGUF metadata still validates registration."""
    name = item['filename']
    if item.get('projector'):
        return False
    # Publishers also use Foo-MTP-Q4_K_M alongside Foo-Q4_K_M for
    # full models with embedded MTP, not separately loadable draft heads.
    plain = re.sub(r'([._-])(?:low[._-])?mtp(?=[._-])', '', name, flags=re.I)
    if plain != name and any(row['filename'].lower() == plain.lower()
                             and not row.get('projector') for row in listing):
        return False
    return bool(item.get('mtp') or DRAFT.search(name))


def _install_source(source: str, revision: str) -> tuple[str, str, str | None]:
    """Normalize a repository or an ordinary Hugging Face file-page URL."""
    source = source.strip()
    filename = None
    if '://' in source:
        try:
            url = urlsplit(source)
            if url.scheme != 'https' or url.hostname != 'huggingface.co' or url.username or url.password or url.port:
                raise ValueError
        except ValueError as exc:
            raise AppError('validation', 'Use an HTTPS huggingface.co model repository or file URL.') from exc
        parts = unquote(url.path).strip('/').split('/')
        if len(parts) < 2:
            raise AppError('validation', 'The Hugging Face URL must include owner/repository.')
        source = '/'.join(parts[:2])
        if len(parts) > 2:
            if len(parts) < 4 or parts[2] not in ('tree', 'blob', 'resolve'):
                raise AppError('validation', 'Use a Hugging Face repository, tree, blob or resolve URL.')
            if revision != 'main' and revision != parts[3]:
                raise AppError('validation', 'The URL revision conflicts with the requested revision.')
            revision = parts[3]
            if len(parts) > 4 and (parts[2] in ('blob', 'resolve') or parts[-1].lower().endswith('.gguf')):
                filename = '/'.join(parts[4:])
        query_file = parse_qs(url.query).get('show_file_info', [])
        if query_file:
            if len(query_file) != 1 or (filename and filename != query_file[0]):
                raise AppError('validation', 'The URL specifies conflicting GGUF files.')
            filename = query_file[0]
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*', source):
        raise AppError('validation', 'Use a Hugging Face owner/repository or HTTPS model URL.')
    if filename and (Path(filename).is_absolute() or '..' in Path(filename).parts or '\\' in filename or '\x00' in filename):
        raise AppError('validation', 'The URL must select a file inside the repository.')
    return source, revision, filename


def _shard_files(selected: dict, listing: list[dict]) -> list[dict]:
    filename = selected['filename']
    if not (match := SHARD.fullmatch(Path(filename).name)):
        return [selected]
    count = int(match[3])
    if not 1 <= count <= 999:
        raise AppError('validation', 'Unsupported GGUF shard count.')
    names = [str(Path(filename).with_name(f'{match[1]}-{index:05d}-of-{count:05d}.gguf')) for index in range(1, count + 1)]
    entries = {item['filename']: item for item in listing}
    if any(name not in entries for name in names):
        raise AppError('validation', 'Repository does not contain every required GGUF shard.')
    return [entries[name] for name in names]


def _identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


class _VerificationCache:
    """Disposable private SHA256 receipts, bounded independently of model size."""
    def __init__(self, root):
        self.path = root / 'verification-cache.json'

    def read(self):
        try:
            with os.fdopen(os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'r') as handle:
                info = os.fstat(handle.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_mode & 0o077 or info.st_size > 1024 * 1024):
                    return {}
                data = json.load(handle)
            if (not isinstance(data, dict) or data.get('version') != 1
                    or not isinstance(data.get('receipts'), dict) or len(data['receipts']) > 1024):
                return {}
            return data['receipts']
        except (OSError, ValueError, RecursionError):
            return {}

    @staticmethod
    def key(path, item):
        return hashlib.sha256(json.dumps([str(path.resolve()), item.get('sha256'), item['size']]).encode()).hexdigest()

    def matches(self, path, item, identity):
        return bool(item.get('sha256')) and self.read().get(self.key(path, item)) == identity

    def save(self, path, item, identity):
        if not item.get('sha256'):
            return
        try:
            receipts = self.read()
            key = self.key(path, item)
            receipts.pop(key, None)
            receipts[key] = identity
            receipts = dict(list(receipts.items())[-1024:])
            _atomic(self.path, {'version': 1, 'receipts': receipts})
        except (OSError, AppError):
            # Verification still succeeded. A read-only/broken cache only costs time.
            pass


def _verify_cached(path: Path, item: dict, receipts=None, force_verify=False, identity_out=None) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise AppError('permission', 'Refusing a cached model symlink.')
    if not stat.S_ISREG(info.st_mode) or (item['size'] and info.st_size != item['size']):
        raise AppError('storage', 'An existing completed download has the wrong size; move it aside before retrying.')
    before = _identity(info)
    if receipts and not force_verify and receipts.matches(path, item, before):
        if _identity(path.lstat()) != before:
            raise AppError('storage', 'Model changed during verification; retry when it is no longer being modified.')
        return True
    if item.get('sha256'):
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as handle:
            if _identity(os.fstat(handle.fileno())) != before:
                raise AppError('storage', 'Model changed during verification; retry when it is no longer being modified.')
            checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
            if _identity(os.fstat(handle.fileno())) != before:
                raise AppError('storage', 'Model changed during verification; retry when it is no longer being modified.')
        if checksum != item['sha256']:
            raise AppError('storage', 'An existing download failed checksum verification; move it aside before retrying.')
    if item['filename'].lower().endswith('.gguf'):
        gguf_metadata(path)
    if _identity(path.lstat()) != before:
        raise AppError('storage', 'Model changed during verification; retry when it is no longer being modified.')
    if receipts:
        receipts.save(path, item, before)
    if identity_out is not None:
        identity_out.append(before)
    return True


def _private_root(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = root.lstat()
    if not stat.S_ISDIR(mode.st_mode) or mode.st_uid != os.getuid() or mode.st_mode & 0o077:
        raise AppError('permission', 'Model registry directory must be private (mode 700) and owned by you.')


def _atomic(path: Path, data):
    if path.is_symlink():
        raise AppError('permission', 'Refusing to overwrite a registry symlink.')
    fd, temporary = tempfile.mkstemp(prefix='.models-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def gguf_metadata(path: Path) -> dict:
    """Read header metadata only; never map tensors or deserialize executable objects."""
    scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    limit = min(path.stat().st_size, 256 * 1024 * 1024)
    with path.open('rb') as handle:
        def read(n):
            if n < 0 or handle.tell() + n > limit:
                raise AppError('parser_limit', 'GGUF metadata is truncated or exceeds the 256 MiB inspection limit.')
            value = handle.read(n)
            if len(value) != n:
                raise AppError('validation', 'Truncated GGUF metadata.')
            return value
        def number(code):
            return struct.unpack('<' + code, read(struct.calcsize(code)))[0]
        def string(keep=True):
            n = number('Q')
            if n > 32 * 1024 * 1024:
                raise AppError('parser_limit', 'A GGUF metadata string exceeds 32 MiB.')
            raw = read(n)
            return raw.decode('utf-8', errors='replace') if keep else None
        def value(kind, keep=True):
            if kind in scalar:
                val = number(scalar[kind])
                return val if keep else None
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, count = number('I'), number('Q')
                if count > 2_000_000 or element == 9:
                    raise AppError('parser_limit', 'Unsupported oversized/nested GGUF metadata array.')
                if element in scalar:
                    n = count * struct.calcsize(scalar[element])
                    if handle.tell() + n > limit:
                        raise AppError('parser_limit', 'GGUF metadata array exceeds inspection limit.')
                    handle.seek(n, 1)
                else:
                    for _ in range(count):
                        value(element, False)
                return None
            raise AppError('validation', 'Unknown GGUF metadata field type.')
        if read(4) != b'GGUF' or number('I') not in (2, 3):
            raise AppError('validation', 'Select a valid GGUF v2/v3 model file.')
        tensors, fields = number('Q'), number('Q')
        if fields > 100000 or tensors > 1000000:
            raise AppError('parser_limit', 'GGUF header count exceeds inspection limits.')
        metadata = {}
        for _ in range(fields):
            key = string()
            keep = (key.startswith('general.') or key.endswith('.context_length') or key.endswith('.parameter_count') or key.endswith('.nextn_predict_layers')
                    or key.endswith(('.block_count', '.embedding_length', '.attention.head_count', '.attention.head_count_kv', '.attention.key_length', '.attention.value_length', '.attention.sliding_window'))
                    or key in ('tokenizer.chat_template', 'clip.has_vision_encoder', 'split.count', 'split.no'))
            result = value(number('I'), keep)
            if keep and result is not None:
                metadata[key] = result
        return metadata


def _main_file(path: Path):
    try:
        info = path.stat()
    except OSError as exc:
        raise AppError('validation', 'Model file is missing or unreadable.') from exc
    if not stat.S_ISREG(info.st_mode) or path.suffix.lower() != '.gguf':
        raise AppError('unsupported', 'Baseline inference requires a regular GGUF file. Safetensors/checkpoints need an appropriate supported GGUF conversion.')
    if re.search(r'(?:mmproj|projector)', path.name, re.I):
        raise AppError('validation', 'This file is a multimodal projector. Select its matching main model and supply the projector separately.')
    metadata = gguf_metadata(path)
    arch = str(metadata.get('general.architecture', ''))
    embedded = isinstance(metadata.get(arch + '.nextn_predict_layers'), int) and metadata.get(arch + '.nextn_predict_layers', 0) > 0
    if (DRAFT.search(path.name) and not embedded) or arch.endswith(('_mtp', '-mtp')) or arch == 'gemma4-assistant' or metadata.get('general.type') in ('mtp', 'draft'):
        raise AppError('validation', 'An MTP/draft head is a companion file, not a main language model.')
    if metadata.get('general.architecture') in ('clip', 'projector') or metadata.get('clip.has_vision_encoder'):
        raise AppError('validation', 'A projector cannot be registered as a language model.')
    return metadata


def validate_model(path: Path, projector_path=None, mtp_path=None) -> tuple[dict, list[Path]]:
    path = path.expanduser().resolve()
    metadata = _main_file(path)
    files = [path]
    if match := SHARD.fullmatch(path.name):
        count = int(match[3])
        if not 1 <= count <= 999:
            raise AppError('validation', 'Unsupported GGUF shard count.')
        files = [path.with_name(f'{match[1]}-{index:05d}-of-{count:05d}.gguf') for index in range(1, count + 1)]
        for index, shard in enumerate(files):
            shard_metadata = _main_file(shard)
            if shard_metadata.get('split.count', count) != count or shard_metadata.get('split.no', index) != index:
                raise AppError('validation', 'GGUF shard metadata does not match its filename/order.')
            if shard_metadata.get('general.architecture') != metadata.get('general.architecture'):
                raise AppError('validation', 'GGUF shards belong to different model architectures.')
        if int(match[2]) != 1:
            raise AppError('validation', 'Register the first GGUF shard (-00001-of-...) after downloading every shard.')
    elif metadata.get('split.count', 1) > 1:
        raise AppError('validation', 'Split GGUF model needs its complete conventionally named shard set.')
    if projector_path:
        projector = Path(projector_path).expanduser().resolve()
        if not projector.is_file() or projector.suffix.lower() != '.gguf':
            raise AppError('validation', 'The selected multimodal projector is missing or is not a GGUF file.')
        pm = gguf_metadata(projector)
        if pm.get('general.architecture') not in (None, 'clip', 'projector') and not pm.get('clip.has_vision_encoder'):
            raise AppError('validation', 'The selected vision companion has language-model metadata, not projector metadata.')
        # Matching architecture is runtime-verified; filenames alone cannot establish it.
    if mtp_path:
        if Path(mtp_path).expanduser().resolve() == path:
            raise AppError('validation', 'Select a separate existing MTP GGUF head.')
        validate_mtp(mtp_path, metadata)
    return metadata, files


def validate_mtp(path, main_metadata) -> dict:
    """Reject clearly wrong companion roles; actual tensor matching requires loading."""
    head = Path(path).expanduser().resolve()
    if not head.is_file() or head.suffix.lower() != '.gguf':
        raise AppError('validation', 'Select a separate existing MTP GGUF head.')
    metadata = gguf_metadata(head)
    arch = str(metadata.get('general.architecture', ''))
    main_arch = str(main_metadata.get('general.architecture', ''))
    if arch in ('clip', 'projector') or metadata.get('clip.has_vision_encoder'):
        raise AppError('validation', 'A vision projector cannot be used as an MTP head.')
    companion_arch = 'gemma4' if arch == 'gemma4-assistant' else re.sub(r'[_-]mtp$', '', arch)
    if arch and main_arch and companion_arch != main_arch:
        raise AppError('validation', 'The MTP head and main model declare different architectures.')
    if not (DRAFT.search(head.name) or arch.endswith(('_mtp', '-mtp')) or arch == 'gemma4-assistant' or metadata.get('general.type') in ('mtp', 'draft') or metadata.get(arch + '.nextn_predict_layers', 0)):
        raise AppError('validation', 'The companion is not identified as an MTP/draft head; select the matching head GGUF.')
    return metadata


class ModelLibrary:
    def __init__(self, root: Path, cache_dir: Path | str | None = None):
        self.root = Path(root).expanduser().absolute()
        _private_root(self.root)
        self.registry = self.root / 'models.json'
        self._verification = _VerificationCache(self.root)
        self.cache_dir = Path(cache_dir or os.environ.get('HPC_LLM_MODEL_CACHE') or self.root / 'downloads').expanduser().absolute()

    def list(self) -> list[ModelSpec]:
        if not self.registry.exists():
            return []
        if self.registry.is_symlink() or self.registry.stat().st_size > 4 * 1024 * 1024:
            raise AppError('permission', 'Unsafe model registry file.')
        try:
            return [ModelSpec.model_validate(item) for item in json.loads(self.registry.read_text())]
        except (ValueError, OSError) as exc:
            raise AppError('storage', 'Model registry is unreadable; preserve it before restoring from backup.') from exc

    def register(self, path, **kwargs) -> ModelSpec:
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            candidates = sorted(p for p in path.glob('*.gguf') if not re.search('mmproj|projector', p.name, re.I) and not DRAFT.search(p.name)
                and (not SHARD.fullmatch(p.name) or SHARD.fullmatch(p.name)[2] == '00001'))
            if not candidates:
                raise AppError('unsupported', 'No main GGUF files found. Safetensors/checkpoint folders require a supported GGUF conversion.')
            if len(candidates) > 500:
                raise AppError('parser_limit', 'Choose a smaller model directory (at most 500 model entries).')
            # Explicit directory registration registers all complete variants without copying weights.
            registered = [self.register(candidate, **kwargs) for candidate in candidates]
            return registered[0]
        models = self.list()
        old = next((m for m in models if m.path == str(path)), None)
        projector = kwargs.pop('projector_path', None) or (old.projector_path if old else None)
        mtp = kwargs.pop('mtp_path', None) or (old.mtp_path if old else None)
        metadata, files = validate_model(path, projector, mtp)
        if old:
            for key in ('name', 'repo_id', 'revision'):
                if not kwargs.get(key):
                    kwargs[key] = getattr(old, key)
        layers = metadata.get(str(metadata.get('general.architecture', '')) + '.nextn_predict_layers', 0)
        if not isinstance(layers, int) or layers < 0:
            raise AppError('validation', 'GGUF MTP layer count must be a nonnegative integer.')
        quant = QUANT.search(path.stem)
        template = metadata.get('tokenizer.chat_template', '')
        thinking = 'enable_thinking' if 'enable_thinking' in template else 'unknown'
        reasoning_efforts, reasoning_default = inspect_reasoning_template(template)
        model = ModelSpec(id=old.id if old else hashlib.sha256(str(path).encode()).hexdigest()[:32], name=kwargs.pop('name', None) or metadata.get('general.name') or path.stem,
            path=str(path), projector_path=str(Path(projector).expanduser().resolve()) if projector else None,
            mtp_path=str(Path(mtp).expanduser().resolve()) if mtp else None,
            mtp_layers=layers,
            memory_metadata={k: v for k, v in metadata.items() if k == "general.architecture" or k.endswith((".block_count", ".embedding_length", ".attention.head_count", ".attention.head_count_kv", ".attention.key_length", ".attention.value_length", ".attention.sliding_window"))},
            size_bytes=sum(p.stat().st_size for p in files), quantization=quant[1].upper() if quant else None,
            total_parameters=metadata.get('general.parameter_count'),
            supported_context=next((int(v) for k, v in metadata.items() if k.endswith('.context_length') and isinstance(v, (int, float))), None),
            thinking=thinking, reasoning_efforts=reasoning_efforts, reasoning_default=reasoning_default, capability_provenance='GGUF metadata/template inspected; runtime compatibility and projector match require loading', **kwargs)
        if old:
            model.defaults = old.defaults
        _atomic(self.registry, [m.model_dump() for m in models if m.id != model.id] + [model.model_dump()])
        return model

    def save_defaults(self, model_id, settings):
        models = self.list()
        model = next((m for m in models if m.id == model_id), None)
        if model is None:
            raise AppError('validation', 'Model is not registered.')
        validated = settings if isinstance(settings, InferenceSettings) else InferenceSettings.model_validate(settings)
        model.defaults = validated.model_dump()
        _atomic(self.registry, [m.model_dump() for m in models])

    def list_remote(self, repo_id, revision='main', include_files=False):
        return list_remote(repo_id, revision, include_files=include_files)

    def list_install_choices(self, source, revision='main') -> dict:
        """List installable variants for an ambiguity chooser, without downloading weights."""
        repo_id, revision, filename = _install_source(source, revision)
        listing = self.list_remote(repo_id, revision)
        mains = [item for item in listing if not item.get('projector') and not _remote_head(item, listing)
                 and (not SHARD.fullmatch(Path(item['filename']).name)
                      or SHARD.fullmatch(Path(item['filename']).name)[2] == '00001')]
        return {'repo_id': repo_id, 'revision': listing[0]['revision'] if listing else revision,
                'models': mains, 'projectors': [item for item in listing if item.get('projector')],
                'mtp_heads': [item for item in listing if _remote_head(item, listing)]}

    def _target(self, model_id):
        if model_id is None:
            return None
        models = self.list()
        target = next((model for model in models if model.id == model_id), None)
        if target is None:
            canonical = str(Path(model_id).expanduser().resolve())
            matches = [model for model in models if model.path == canonical or model.name == model_id]
            if len(matches) > 1:
                raise AppError('validation', 'Model name is ambiguous; use its registered path or ID.')
            target = matches[0] if matches else None
        if target is None:
            raise AppError('validation', 'Model is not registered.')
        return target

    def plan_install(self, source=None, quant=None, revision=None, projector=None, mtp='none', model_id=None, force_verify=False) -> dict:
        """Pin a new installation or a companion-only upgrade of a registered model."""
        target = self._target(model_id)
        source = source or (target.repo_id if target else None)
        if not source:
            raise AppError('validation', 'This local model has no repository provenance; supply its Hugging Face repository explicitly.')
        revision = revision or (target.revision if target else None) or 'main'
        repo_id, revision, filename = _install_source(source, revision)
        download_only = bool(filename and not filename.lower().endswith('.gguf'))
        if download_only:
            if target or projector not in (None, 'auto', 'none') or mtp not in (None, 'auto', 'none'):
                raise AppError('validation', 'Download non-GGUF files separately from registered-model companion upgrades.')
            listing = self.list_remote(repo_id, revision, include_files=True)
            selected = next((item for item in listing if item['filename'] == filename), None)
            if selected is None:
                raise AppError('validation', 'The file is not present in this repository revision.')
            destination = self.cache_dir / repo_id.replace('/', '--') / selected['revision']
            cached = _verify_cached(destination / filename, selected, self._verification, force_verify)
            return {'repo_id': repo_id, 'revision': selected['revision'], 'filename': filename,
                    'download_only': True, 'model_id': None, 'model_path': None,
                    'projector_filename': None, 'mtp_filename': None, 'files': [selected],
                    'size_bytes': selected['size'], 'destination': str(destination),
                    'cached_bytes': (destination / filename).stat().st_size if cached else 0,
                    'force_verify': bool(force_verify)}
        listing = self.list_remote(repo_id, revision)
        mains = [item for item in listing if not item.get('projector') and not _remote_head(item, listing)]
        # An explicitly selected filename may be an embedded-MTP main despite its
        # name; downloaded GGUF metadata determines its role before registration.
        explicit = filename or (quant if quant and quant.lower().endswith('.gguf') else None)
        if explicit:
            extra = next((item for item in listing if item['filename'] == explicit and not item.get('projector')), None)
            if extra and extra not in mains:
                mains.append(extra)
        if filename:
            selected = next((item for item in mains if item['filename'] == filename), None)
            if selected is None:
                raise AppError('validation', 'The URL must select a main GGUF file from this repository revision.')
            candidates = [_shard_files(selected, listing)[0]]
        else:
            candidates = [item for item in mains if not SHARD.fullmatch(Path(item['filename']).name)
                          or SHARD.fullmatch(Path(item['filename']).name)[2] == '00001']
        if target and not quant and not filename:
            matches = [item for item in candidates if Path(item['filename']).name == Path(target.path).name]
            if matches:
                candidates = matches
        if quant:
            exact = [item for item in candidates if item['filename'] == quant]
            pattern = re.compile(r'(?:^|[._-])' + re.escape(quant) + r'(?:[._-]|$)', re.I)
            candidates = exact or [item for item in candidates if pattern.search(Path(item['filename']).stem)]
        if len(candidates) != 1:
            choices = ', '.join(item['filename'] for item in candidates or mains if not SHARD.fullmatch(Path(item['filename']).name)
                                or SHARD.fullmatch(Path(item['filename']).name)[2] == '00001')
            raise AppError('validation', 'Choose one model using --quant QUANT_OR_FILENAME. Available GGUF files: ' + (choices or '(none)'))
        selected = candidates[0]
        main_files = _shard_files(selected, listing)
        if target:
            self._verify_target(target, main_files, force_verify)
        projectors = [item for item in listing if item.get('projector')]
        main_names = {item['filename'] for item in main_files}
        heads = [item for item in listing if item['filename'] not in main_names
                 and _remote_head(item, listing)]
        companion = None
        projector = projector if projector is not None else ('none' if target else 'auto')
        if projector == 'auto':
            if len(projectors) == 1:
                companion = projectors[0]
            elif projectors:
                f16 = [item for item in projectors if Path(item['filename']).name.lower() == 'mmproj-f16.gguf']
                same_family = all(re.fullmatch(r'mmproj-(?:f16|bf16|f32|q\d(?:_[a-z0-9]+)*)\.gguf', item['filename'], re.I)
                                  for item in projectors)
                if len(f16) == 1 and same_family:
                    companion = f16[0]
                else:
                    raise AppError('validation', 'Multiple vision projectors are available; specify --projector FILENAME (or none): '
                                   + ', '.join(item['filename'] for item in projectors))
        elif projector and projector != 'none':
            companion = next((item for item in projectors if item['filename'] == projector), None)
            if companion is None:
                raise AppError('validation', 'Choose an exact projector filename from this repository revision.')
        head = None
        if mtp == 'auto':
            if len(heads) > 1:
                raise AppError('validation', 'Multiple MTP heads are available; specify --mtp FILENAME: ' + ', '.join(item['filename'] for item in heads))
            head = heads[0] if heads else None
        elif mtp and mtp != 'none':
            head = next((item for item in heads if item['filename'] == mtp), None)
            if head is None:
                raise AppError('validation', 'Choose an exact MTP head filename from this repository revision.')
        chosen = ([] if target else main_files) + ([companion] if companion else []) + ([head] if head else [])
        pinned = selected['revision']
        destination = self.cache_dir / repo_id.replace('/', '--') / pinned
        cached_bytes = sum((destination / item['filename']).stat().st_size for item in chosen
                           if _verify_cached(destination / item['filename'], item, self._verification, force_verify))
        return {'repo_id': repo_id, 'revision': pinned, 'filename': selected['filename'],
                'model_id': target.id if target else None, 'model_path': target.path if target else None,
                'projector_filename': companion['filename'] if companion else None,
                'mtp_filename': head['filename'] if head else None, 'files': chosen,
                'size_bytes': sum(item['size'] for item in chosen), 'destination': str(destination),
                'cached_bytes': cached_bytes, 'force_verify': bool(force_verify)}

    def _verify_target(self, target, main_files, force_verify=False):
        _, paths = validate_model(Path(target.path))
        if len(paths) != len(main_files):
            raise AppError('validation', 'The selected repository variant does not match the registered main model shards.')
        for path, item in zip(paths, main_files):
            # Filename similarity is not compatibility evidence. Require exact bytes.
            if not item.get('sha256'):
                raise AppError('validation', 'Repository lacks a main-model checksum; cannot verify a companion upgrade safely. Register downloaded companions explicitly instead.')
            try:
                if not _verify_cached(path, item, self._verification, force_verify):
                    raise AppError('storage', 'Registered model disappeared during verification.')
            except AppError as exc:
                raise AppError('validation', 'The selected repository main model does not match the registered weights; choose their original repository/revision.') from exc

    def execute_install(self, plan, cancel=None, progress=None) -> ModelSpec | Path:
        # Re-resolve only the immutable revision and explicit filenames, never a branch.
        pinned = plan['revision']
        if not re.fullmatch(r'[0-9a-f]{40,64}', pinned):
            raise AppError('validation', 'Installation plan must use an immutable revision.')
        source = (f"https://huggingface.co/{plan['repo_id']}/resolve/{pinned}/{quote(plan['filename'])}"
                  if plan.get('download_only') else plan['repo_id'])
        verified = self.plan_install(source, quant=plan['filename'], revision=pinned,
            projector=plan.get('projector_filename') or 'none', mtp=plan.get('mtp_filename') or 'none', model_id=plan.get('model_id'))
        if verified['files'] != plan['files'] or verified['model_path'] != plan.get('model_path'):
            raise AppError('validation', 'The installation plan changed; preview it again before downloading.')
        destination = Path(verified['destination'])
        _private_root(destination)
        for item in verified['files']:
            self._download_one(verified['repo_id'], item, destination, cancel, progress)
        if cancel and (cancel() if callable(cancel) else cancel.is_set()):
            raise AppError('network', 'Download cancelled; the model registration was not changed.')
        if verified.get('download_only'):
            return destination / verified['filename']
        target = self._target(plan.get('model_id'))
        if target and target.path != plan['model_path']:
            raise AppError('validation', 'Registered model changed during download; preview the upgrade again.')
        if target:
            listing = self.list_remote(verified['repo_id'], pinned)
            selected = next((item for item in listing if item['filename'] == verified['filename']), None)
            if selected is None:
                raise AppError('validation', 'The pinned main model is no longer listed.')
            self._verify_target(target, _shard_files(selected, listing))
        kwargs = {} if target else {'repo_id': verified['repo_id'], 'revision': pinned}
        for key in ('projector', 'mtp'):
            if verified.get(key + '_filename'):
                kwargs[key + '_path'] = destination / verified[key + '_filename']
        return self.register(target.path if target else destination / verified['filename'], **kwargs)

    def install(self, source=None, quant=None, revision=None, projector=None, cancel=None, progress=None, mtp='none', model_id=None, force_verify=False) -> ModelSpec | Path:
        return self.execute_install(self.plan_install(source, quant, revision, projector, mtp, model_id, force_verify), cancel, progress)

    def download(self, repo_id, filename, revision, cancel=None, progress=None, projector_filename=None, mtp_filename=None):
        if re.search('mmproj|projector', filename, re.I):
            raise AppError('validation', 'Select a main model; a projector or MTP head is a companion file, not a language model.')
        return self.install(repo_id, quant=filename, revision=revision, projector=projector_filename or 'none',
                            mtp=mtp_filename or 'none', cancel=cancel, progress=progress)

    def _download_one(self, repo_id, item, destination, cancel, progress):
        filename = item['filename']
        path = destination / filename
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != self.root.parent):
            raise AppError('permission', 'Refusing a download destination containing a symlink.')
        if _verify_cached(path, item, self._verification):
            return
        partial = path.with_suffix(path.suffix + '.part')
        if partial.is_symlink():
            raise AppError('permission', 'Refusing a symlink partial download.')
        offset = partial.stat().st_size if partial.exists() else 0
        url = f'https://huggingface.co/{repo_id}/resolve/{item["revision"]}/{quote(filename)}?download=true'
        request = Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urlopen(request, timeout=30) as response:
                resumed = offset and response.status == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                if response.status == 206 and not resumed:
                    raise AppError('network', 'Download server returned an unexpected byte range.')
                if not resumed:
                    offset = 0
                flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_APPEND if resumed else os.O_TRUNC)
                with os.fdopen(os.open(partial, flags, 0o600), 'wb') as output:
                    while True:
                        cancelled = cancel and (cancel() if callable(cancel) else cancel.is_set())
                        if cancelled:
                            raise AppError('network', 'Download cancelled; partial bytes are retained for an explicit resume.')
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        offset += len(chunk)
                        if item['size'] and offset > item['size']:
                            raise AppError('network', 'Download exceeded its declared size.')
                        if progress:
                            progress(offset, item['size'])
                    output.flush()
                    os.fsync(output.fileno())
            if item['size'] and offset != item['size']:
                raise AppError('network', 'Download incomplete; rerun to resume.')
            # The same verifier handles resumed bytes and records successful validation.
            # Verify under the partial name, then seed the final name after rename.
            verified_identity = []
            _verify_cached(partial, item, force_verify=True, identity_out=verified_identity)
            before = verified_identity[0]
            if _identity(partial.lstat()) != before:
                raise AppError('storage', 'Model changed before publishing the verified download.')
            os.replace(partial, path)
            after = _identity(path.lstat())
            # Rename may update ctime; the inode, size and mtime must remain identical.
            if before[:4] != after[:4]:
                raise AppError('storage', 'Model changed while publishing the verified download.')
            # A changed ctime cannot safely be attributed solely to rename.
            # Leave the final path uncached in that case; next use verifies it.
            if before == after:
                self._verification.save(path, item, after)
        except (OSError, URLError) as exc:
            raise AppError('network', 'Model download failed; partial bytes are retained. Check download egress, disk quota, and retry explicitly.') from exc


def list_remote(repo_id, revision='main', include_files=False):
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*', repo_id) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', revision):
        raise AppError('validation', 'Use a Hugging Face owner/repository and revision name or commit hash.')
    try:
        with urlopen(f'https://huggingface.co/api/models/{repo_id}/revision/{quote(revision)}?blobs=true', timeout=20) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise AppError('parser_limit', 'Repository listing is too large.')
        metadata = json.loads(raw)
        pinned = metadata.get('sha', '')
        if not re.fullmatch(r'[0-9a-f]{40,64}', pinned):
            raise AppError('provider', 'Repository did not return an immutable commit identifier.')
        if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision, re.I) and pinned.lower() != revision.lower():
            raise AppError('provider', 'Repository metadata does not match the requested immutable revision.')
        result = []
        for entry in metadata.get('siblings', []):
            filename = entry.get('rfilename', '')
            if (filename and (include_files or filename.lower().endswith('.gguf'))
                    and not Path(filename).is_absolute() and '..' not in Path(filename).parts
                    and '\\' not in filename and '\x00' not in filename):
                lfs = entry.get('lfs') or {}
                result.append({'filename': filename, 'revision': pinned, 'size': lfs.get('size') or entry.get('size') or 0,
                    'sha256': lfs.get('sha256'), 'projector': bool(re.search('mmproj|projector', filename, re.I)), 'mtp': bool(DRAFT.search(filename))})
        return result
    except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
        raise AppError('network', 'Unable to list the Hugging Face repository. Check its name, visibility, revision and network access.') from exc


def download(repo_id, filename, revision, cancel=None, progress=None, projector_filename=None, mtp_filename=None):
    root = Path(os.environ.get('HPC_LLM_HOME', Path.cwd() / 'state')) / 'models'
    return ModelLibrary(root).download(repo_id, filename, revision, cancel, progress, projector_filename, mtp_filename)
