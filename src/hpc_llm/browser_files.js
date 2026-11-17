'use strict';
const $ = id => document.getElementById(id);
const base = location.pathname.replace(/\/$/, '');
let currentTask = null, pending = null, shownApproval = null, pollTimer = null, lastTask = null;
const status = message => { $('status').textContent = message; };
async function api(route, data) {
  const options = {credentials:'same-origin',cache:'no-store'};
  if (data !== undefined) {options.method='POST';options.headers={'Content-Type':'application/json'};options.body=JSON.stringify(data);}
  const response = await fetch(base + route, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed ('+response.status+')');
  return result;
}
const selection = () => Number($('root').value);
function downloaded(path) {
  $('download').hidden = false;
  $('download').href = base + '/download?root=' + selection() + '&path=' + encodeURIComponent(path);
  $('download').download = path.split('/').pop();
}
async function browse() {
  const result = await api('/list',{root:selection(),path:$('directory').value});
  $('directory').value=result.path;
  $('listing').replaceChildren();
  for (const entry of result.entries) {
    const button=document.createElement('button');
    button.textContent=(entry.directory ? 'Directory: ' : 'File: ') + entry.name;
    button.onclick=()=>guard(async()=>{
      if(entry.directory) {$('directory').value=entry.path;await browse();}
      else {$('destination').value=entry.path;await read();}
    });
    $('listing').append(button);
  }
  status(result.truncated ? 'Listing limited; enter an exact path to a child directory.' : result.entries.length+' entries. Hidden files, links and application secrets are excluded.');
}
async function read() {
  const result=await api('/read',{root:selection(),path:$('destination').value});
  $('content').value=result.content;$('destination').value=result.path;downloaded(result.path);status('Read '+result.path+' ('+result.size+' bytes).');
}
function showPreview(value, source) {
  pending={value,source};
  $('preview-title').textContent=value.operation==='mkdir_workspace' ? 'Approve directory creation' : (value.overwrite ? 'Approve file overwrite' : 'Approve file creation');
  $('preview-path').textContent=value.path;
  $('preview-content').textContent=value.operation==='mkdir_workspace' ? '(New empty directory)' : value.content;
  $('preview-diff').textContent=value.diff || '(No content diff)';
  $('approve').textContent=value.operation==='mkdir_workspace' ? 'Approve directory' : 'Approve and save';
  if(!$('preview').open) $('preview').showModal();
}
async function preview(operation) {
  const args=operation==='mkdir_workspace' ? {path:$('new-directory').value} : {path:$('destination').value,content:$('content').value};
  showPreview(await api('/preview',{root:selection(),operation,arguments:args}),'manual');
}
async function decide(approved) {
  if(!pending) return;
  const {value,source}=pending;pending=null;$('preview').close();
  if(source==='task') {
    await api('/decision',{id:currentTask,approval_id:value.id,approved});
    await poll();
  } else {
    const result=await api('/approve',{id:value.id,approved});
    if(result.declined) {status('Declined. Nothing was saved.');return;}
    status(result.saved ? 'Saved '+result.path+' ('+result.size+' bytes).' : 'Created directory '+result.path);
    if(result.saved) downloaded(result.path);
    await browse();
    status(result.saved ? 'Saved '+result.path+' ('+result.size+' bytes).' : 'Created directory '+result.path);
  }
}
async function poll() {
  if(!currentTask) return;
  clearTimeout(pollTimer);
  const task=await api('/tasks/'+encodeURIComponent(currentTask));lastTask=task;
  $('activity').textContent='Status: '+task.status+'\n\n'+task.events.map(event=>event.name+' → '+event.path+'\n'+event.status+(event.result ? '\n'+JSON.stringify(event.result,null,2) : '')).join('\n\n')+'\n\n'+task.answer;
  if(task.pending && shownApproval!==task.pending.id) {shownApproval=task.pending.id;showPreview(task.pending,'task');}
  if(!task.pending && pending && pending.source==='task') {pending=null;$('preview').close();}
  const active=['running','approval'].includes(task.status);
  $('cancel').disabled=!active;$('run').disabled=active;$('root').disabled=active;
  if(active) pollTimer=setTimeout(()=>guard(poll),700);
  else {currentTask=null;await browse();status('File task '+task.status+'. See its activity and result below.');}
}
async function guard(fn) {try {await fn();} catch(error) {status(error.message);}}
$('browse').onclick=()=>guard(browse);
$('read').onclick=()=>guard(read);
$('save').onclick=()=>guard(()=>preview('write_workspace'));
$('mkdir').onclick=()=>guard(()=>preview('mkdir_workspace'));
$('approve').onclick=()=>guard(()=>decide(true));
$('deny').onclick=()=>guard(()=>decide(false));
$('preview').addEventListener('cancel',event=>{event.preventDefault();guard(()=>decide(false));});
$('root').onchange=()=>guard(async()=>{$('directory').value=$('root').selectedOptions[0].textContent;$('download').hidden=true;await browse();});
$('up').onclick=()=>guard(async()=>{const root=$('root').selectedOptions[0].textContent;const value=$('directory').value.replace(/\/$/,'');$('directory').value=value===root ? root : value.slice(0,value.lastIndexOf('/'))||root;await browse();});
$('run').onclick=()=>guard(async()=>{const result=await api('/tasks',{root:selection(),prompt:$('prompt').value});currentTask=result.id;shownApproval=null;await poll();});
$('cancel').onclick=()=>guard(async()=>{if(currentTask){await api('/cancel',{id:currentTask});await poll();}});
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({task:lastTask,prompt:$('prompt').value},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='file-task.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
guard(async()=>{
  const info=await api('/info');
  for(let index=0;index<info.roots.length;index++){const option=document.createElement('option');option.value=index;option.textContent=info.roots[index];$('root').append(option);}
  if(!info.roots.length){status('No workspace is enabled. Select a workspace in the launcher before submitting a new session.');for(const id of ['browse','read','save','mkdir','run','up']) $(id).disabled=true;return;}
  $('directory').value=info.roots[0];
  $('content').maxLength=info.limit;
  await browse();
  status('Selected cluster workspace. Limit: '+info.limit+' bytes/file, '+info.max_tool_calls+' tool calls/task, '+info.timeout_seconds+' seconds/task (including approval).'+(info.auto_write ? ' Automatic model writes are enabled within this workspace.' : ' Model writes require preview approval.'));
});
