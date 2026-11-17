"""Exercise CLI configuration selection and resource precedence through launcher."""
import asyncio
from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from textual.widgets import Input, Select
from hpc_llm import cli
from hpc_llm.config import AppConfig, save_config
from hpc_llm.contracts import ModelSpec, ResourceRequest, SessionManifest, SiteProfile
from hpc_llm.ui import LauncherApp, ResourceDialog

class LauncherConfigChecks(unittest.TestCase):
    def test_configured_state_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'HPC_LLM_CONFIG': '', 'HPC_LLM_HOME': ''}):
            root = Path(folder)
            config = root / 'custom.yaml'
            state = root / 'configured-state'
            save_config(config, AppConfig(state_directory=str(state), profiles={'slurm': SiteProfile()}))
            with patch('hpc_llm.cli.check', return_value=0) as check:
                self.assertEqual(cli.main(['--config', str(config), '--check']), 0)
                check.assert_called_once_with(state)
            explicit = root / 'explicit-state'
            with patch('hpc_llm.cli.check', return_value=0) as check:
                self.assertEqual(cli.main(['--config', str(config), '--state-root', str(explicit), '--check']), 0)
                check.assert_called_once_with(explicit)

    def test_snapshot_commands_ignore_live_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            invalid = root / 'broken.yaml'
            invalid.write_text('schema_version: invalid')
            manager = SimpleNamespace(list=lambda: [])
            with patch.dict(os.environ, {'HPC_LLM_CONFIG': str(invalid)}), patch('hpc_llm.cli._manager', return_value=manager) as factory:
                self.assertEqual(cli.main(['--state-root', str(root / 'session-state'), 'sessions']), 0)
                factory.assert_called_once_with(root / 'session-state')

    def test_cli_resources_override_preset_and_edited_choice(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'HPC_LLM_CONFIG': ''}):
            root = Path(folder)
            (root / 'work').mkdir()
            profile = SiteProfile(resources=ResourceRequest(partition='default-gpu', cpus=4),
                resource_presets={'quick': ResourceRequest(partition='preset-gpu', cpus=8, memory_gb=24, walltime='00:20:00')})
            profile.connection.mode = 'local-forward'
            model = ModelSpec(name='fixture', path='/fixture.gguf')
            captured = {}
            class FakeApp:
                def __init__(self, models, sessions, effective, **kwargs):
                    if 'display' in captured:
                        raise AssertionError('Unexpected launcher retry: ' + kwargs.get('error_message', ''))
                    captured['display'] = effective.resources.model_dump()
                    captured['overrides'] = kwargs['resource_overrides']
                def run(self):
                    return {'action': 'start', 'model_id': model.id,
                        'resources': {'partition': 'edited-gpu', 'cpus': 3, 'memory_gb': 40, 'walltime': '00:35:00'}}
            def create(selected_model, resources, selected_profile, settings):
                captured['submitted'] = resources.model_dump()
                return SessionManifest(owner='fixture', directory=str(root / 'session'), model=selected_model, resources=resources, profile=selected_profile)
            manager = SimpleNamespace(inventory=lambda profile: {}, list=lambda: [], create=create, submit=lambda m: m)
            options = cli.parse_args(['launch', '--preset', 'quick', '--cpus', '12', '--walltime', '00:50:00', '--workspace', str(root / 'work')])
            with ExitStack() as stack:
                stack.enter_context(patch('hpc_llm.cli.profile_for', side_effect=lambda _: profile.model_copy(deep=True)))
                stack.enter_context(patch('hpc_llm.cli._library', return_value=SimpleNamespace(list=lambda: [model])))
                stack.enter_context(patch('hpc_llm.ui.LauncherApp', FakeApp))
                stack.enter_context(patch('hpc_llm.compute_web.saved_origin', return_value=''))
                stack.enter_context(patch('hpc_llm.compute_web.prepare_browser'))
                stack.enter_context(patch('hpc_llm.cli._wait_ready', side_effect=lambda manager, manifest: manifest))
                stack.enter_context(patch('hpc_llm.cli.attach', return_value=0))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(cli.launcher(manager, root, options=options), 0)
            self.assertEqual((captured['display']['partition'], captured['display']['cpus'], captured['display']['walltime']), ('preset-gpu', 12, '00:50:00'))
            result = captured['submitted']
            self.assertEqual((result['partition'], result['cpus'], result['memory_gb'], result['walltime']), ('edited-gpu', 12, 40, '00:50:00'))
            self.assertEqual(profile.resources.partition, 'default-gpu')
            self.assertEqual(profile.resource_presets['quick'].cpus, 8)

    def test_headless_ui_locks_cli_fields_and_preserves_them_on_preset_reset(self):
        async def exercise():
            profile = SiteProfile(resources=ResourceRequest(cpus=12, walltime='00:50:00'),
                resource_presets={'quick': ResourceRequest(cpus=8, walltime='00:20:00')})
            app = LauncherApp([ModelSpec(name='fixture', path='/fixture.gguf')], [], profile,
                resource_overrides={'cpus': 12, 'walltime': '00:50:00'}, connection_overrides={'mode': 'ood'})
            async with app.run_test(size=(120, 55)) as pilot:
                self.assertTrue(app.query_one('#launch-walltime', Input).disabled)
                self.assertTrue(app.query_one('#launch-connection', Select).disabled)
                dialog = ResourceDialog(profile.resources.model_dump(), profile=profile.model_dump(), overrides=app.resource_overrides)
                app.push_screen(dialog)
                await pilot.pause()
                self.assertTrue(dialog.query_one('#resource-cpus', Input).disabled)
                self.assertTrue(dialog.query_one('#resource-walltime', Input).disabled)
                self.assertFalse(dialog.query_one('#resource-partition', Input).disabled)
                dialog.query_one('#resource-preset', Select).value = 'quick'
                await pilot.pause()
                self.assertEqual(dialog.query_one('#resource-cpus', Input).value, '12')
                self.assertEqual(dialog.query_one('#resource-walltime', Input).value, '00:50:00')
                await pilot.click('#resource-reset')
                self.assertEqual(dialog.query_one('#resource-cpus', Input).value, '12')
                self.assertEqual(dialog.query_one('#resource-walltime', Input).value, '00:50:00')
        asyncio.run(exercise())

if __name__ == '__main__':
    unittest.main()
