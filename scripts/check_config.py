"""Offline configuration, migration and scheduler boundary regression checks."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from hpc_llm.config import AppConfig, load_config, resolve_resources, save_config
from hpc_llm.contracts import AppError, ModelSpec, ResourceRequest, SessionManifest, SiteProfile
from hpc_llm.profiles import ProfileStore
from hpc_llm.schedulers import PBSScheduler, SlurmScheduler

APP = Path(__file__).resolve().parents[1]

class ConfigChecks(unittest.TestCase):
    def test_new_install_defaults_to_slurm(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'HPC_LLM_CONFIG': ''}):
            root = Path(folder)
            store = ProfileStore(root, APP)
            profile = store.configure_install(cache=root / 'cache')
            self.assertEqual((profile.name, profile.scheduler, profile.resources.partition), ('slurm', 'slurm', 'gpu'))
            self.assertEqual(profile.connection.mode, 'local-forward')
            self.assertEqual(store.document().selected_profile, 'slurm')
            self.assertEqual(store.document().profiles['wehi'].scheduler, 'slurm')
            self.assertEqual(store.document().profiles['wehi'].resources.partition, 'gpuq')

    def test_upgrade_preserves_selection_and_offers_slurm(self):
        for name in ('generic', 'wehi', 'custom'):
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / 'config.yaml'
                saved = SiteProfile(name=name, resources=ResourceRequest(partition='my-queue'), workspaces=[str(root / 'work')])
                save_config(path, AppConfig(selected_profile=name, model_cache=str(root / 'cache'), profiles={name: saved}))
                store = ProfileStore(root, APP, path)
                before = path.read_bytes()
                self.assertIn('slurm', [profile.name for profile in store.list()])
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(store.configure_install().name, name)
                selected = store.configure_install(profile='slurm')
                self.assertEqual((selected.name, selected.scheduler), ('slurm', 'slurm'))
                document = load_config(path)
                self.assertEqual(document.selected_profile, 'slurm')
                self.assertEqual(document.profiles[name], saved)
                self.assertEqual(document.model_cache, str(root / 'cache'))
                custom = document.profiles['slurm'].model_copy(deep=True)
                custom.resources.partition = 'custom-slurm-queue'
                custom.workspaces = [str(root / 'selected-workspace')]
                store.save(custom)
                self.assertEqual(store.configure_install(profile='slurm').resources.partition, 'custom-slurm-queue')
                self.assertEqual(store.document().profiles['slurm'], custom)

    def test_old_implicit_generic_config_remains_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('schema_version: 1\nprofiles:\n  generic:\n    resources:\n      partition: saved-queue\n')
            config = load_config(path)
            self.assertEqual(config.selected_profile, 'generic')
            self.assertEqual(config.profiles['generic'].resources.partition, 'saved-queue')

    def test_examples_and_no_evaluation(self):
        for path in (APP / 'examples').glob('config.*.yaml'):
            self.assertEqual(load_config(path).schema_version, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            for data in ('!!python/object/apply:os.system [echo unsafe]',
                'schema_version: 1\nschema_version: 2', 'a: &a {b: c}\nprofiles: *a',
                'schema_version: true\nprofiles: {}', 'schema_version: 2\nprofiles: {}',
                'schema_version: 1\nselected_profile: generic\nprofiles:\n  generic:\n    resources:\n      gpu_count: 2',
                'schema_version: 1\nselected_profile: generic\nprofiles:\n  generic:\n    modules: ["cuda; touch /tmp/unsafe"]'):
                path.write_text(data)
                with self.assertRaises(AppError):
                    load_config(path)

    def test_precedence_without_mutation(self):
        profile = SiteProfile(resources=ResourceRequest(partition='research-v100', cpus=12),
            resource_presets={'quick': ResourceRequest(partition='vis_gpu', cpus=8, walltime='00:10:00')}, default_preset='quick')
        before = profile.model_dump()
        result = resolve_resources(profile, session={'cpus': 16, 'memory_gb': 60}, cli={'cpus': 20, 'memory_gb': None})
        self.assertEqual((result.partition, result.cpus, result.memory_gb, result.walltime), ('vis_gpu', 20, 60, '00:10:00'))
        self.assertEqual(profile.model_dump(), before)
        self.assertEqual(resolve_resources(profile, preset='').partition, 'research-v100')
        for arguments in ({'cli': {'gpu_count': 2}}, {'preset': 'missing'}):
            with self.assertRaises(AppError):
                resolve_resources(profile, **arguments)
        for values in ({'partition': ''}, {'cpus': True}, {'gpu_type': '-bad'}, {'partition': 'x\n#SBATCH --exclusive'}):
            with self.assertRaises(ValidationError):
                ResourceRequest(**values)

    def test_lossless_migration_and_canonical_writes(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'HPC_LLM_CONFIG': ''}):
            root = Path(folder)
            originals = {
                'config.json': json.dumps({'profile': 'mine', 'model_cache': str(root / 'cache')}),
                'profiles/mine.json': json.dumps({'name': 'mine', 'model_cache': 'old-cache', 'resources': {'partition': 'arbitrary-queue'}, 'resource_presets': {'saved': {'partition': 'another-queue'}}}),
                'models/library.json': '{"models":[{"path":"/exact/model.gguf","mtp_path":"/exact/mtp.gguf"}]}',
                'sessions/record.json': '{"history":"retained"}', 'defaults.json': '{"context":8192}'}
            for name, content in originals.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            store = ProfileStore(root, APP)
            self.assertEqual(store.load().resources.partition, 'arbitrary-queue')
            self.assertEqual(store.load().model_cache, str(root / 'cache'))
            for name, content in originals.items():
                self.assertEqual((root / name).read_text(), content)
            for name in ('config.json', 'profiles/mine.json'):
                self.assertEqual((root / 'migration-backup-v1' / name).read_text(), originals[name])
            self.assertEqual(load_config(root / 'config.yaml').profiles['mine'].resource_presets['saved'].partition, 'another-queue')
            (root / 'config.json').write_text('{"profile":"wehi"}')
            self.assertEqual(store.load().name, 'mine')
            store.set_cache(root / 'new-cache')
            self.assertEqual(load_config(root / 'config.yaml').model_cache, str(root / 'new-cache'))
            self.assertEqual((root / 'config.json').read_text(), '{"profile":"wehi"}')
            self.assertEqual((root / 'config.yaml').stat().st_mode & 0o777, 0o600)

    def test_custom_file_and_bad_legacy_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / 'custom.yaml'
            save_config(config, AppConfig(profiles={'slurm': SiteProfile()}))
            with patch.dict(os.environ, {'HPC_LLM_CONFIG': str(config)}):
                store = ProfileStore(root / 'state', APP)
                self.assertEqual(store.config_path, config)
                self.assertEqual(store.config()['profile'], 'slurm')
            state = root / 'legacy'
            state.mkdir()
            old = '{"profile":3}'
            (state / 'config.json').write_text(old)
            with self.assertRaises(AppError):
                ProfileStore(state, APP, state / 'config.yaml').document()
            self.assertEqual((state / 'config.json').read_text(), old)
            self.assertFalse((state / 'config.yaml').exists())

    def test_scheduler_arbitrary_resources_and_literal_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            launcher = root / 'launch'
            launcher.write_text('#!/bin/bash\nprintf "%s" "$CONFIG_LITERAL" > "$2/result"\n')
            launcher.chmod(0o700)
            literal = '$(touch ' + str(root / 'unsafe') + ')'
            resources = ResourceRequest(partition='research_gpu-v2', gpu_type='h200.sxm', cpus=7, memory_gb=91,
                walltime='2-03:04:05', account='proj-a', qos='gpu-long', constraint='a100|h200')
            profile = SiteProfile(runtime_environment={'CONFIG_LITERAL': literal})
            manifest = SessionManifest(owner='user', directory=str(root / 'session'), model=ModelSpec(name='fixture', path='/model'), resources=resources, profile=profile)
            script = SlurmScheduler().render_batch(manifest, launcher, root)
            for expected in ('--partition=research_gpu-v2', '--gres=gpu:h200.sxm:1', '--cpus-per-task=7', '--mem=91G', '--time=2-03:04:05', '--account=proj-a', '--qos=gpu-long', '--constraint=a100|h200'):
                self.assertIn(expected, script)
            subprocess.run(['bash'], input=script, text=True, check=True)
            self.assertEqual((root / 'result').read_text(), literal)
            self.assertFalse((root / 'unsafe').exists())
            manifest.profile = SiteProfile(scheduler='pbs', pbs_gpu_type_resource='gpu_model')
            with self.assertRaises(AppError):
                PBSScheduler().render_batch(manifest, launcher, root)
            manifest.resources = resources.model_copy(update={'qos': '', 'constraint': ''})
            pbs = PBSScheduler().render_batch(manifest, launcher, root)
            self.assertIn('#PBS -q research_gpu-v2', pbs)
            self.assertIn('select=1:ncpus=7:mem=91gb:ngpus=1:gpu_model=h200.sxm', pbs)
            self.assertIn('walltime=51:04:05', pbs)

if __name__ == '__main__':
    unittest.main()
