const API_BASE = '/api';

const $btnRecord  = document.getElementById('btn-record');
const $btnStop    = document.getElementById('btn-stop');
const $btnProcess = document.getElementById('btn-process');
const $spinner    = document.getElementById('spinner');
const $waveform   = document.getElementById('waveform');
const $timer      = document.getElementById('timer');
const $status     = document.getElementById('status');
const $preview    = document.getElementById('preview');
const $hint       = document.getElementById('hint');
const $dashboard  = document.getElementById('btn-dashboard');

let mediaRecorder = null;
let recordedChunks = [];
let stream = null;
let startTime = 0;
let timerInterval = null;
let jobId = null;

async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body && method !== 'GET') opts.body = JSON.stringify(body);
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(function() { return { detail: res.statusText }; });
    throw new Error(err.detail || 'API ' + res.status);
  }
  return res.json();
}

async function uploadBlob(id, blob) {
  const form = new FormData();
  var ext = blob.type.includes('mp4') ? '.mp4' : '.webm';
  form.append('file', blob, 'recording_' + Date.now() + ext);
  const res = await fetch(API_BASE + '/jobs/' + id + '/upload', { method: 'POST', body: form });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

function tickTimer() {
  const s = Math.floor((Date.now() - startTime) / 1000);
  const m = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = (s % 60).toString().padStart(2, '0');
  $timer.textContent = m + ':' + sec;
}

function showBtn(name) {
  $btnRecord.classList.toggle('hidden', name !== 'record');
  $btnStop.classList.toggle('hidden', name !== 'stop');
  $btnProcess.classList.toggle('hidden', name !== 'process');
  $spinner.classList.toggle('hidden', name !== 'spinner');
}

function setStatus(text, cls) {
  $status.textContent = text;
  $status.className = 'status' + (cls ? ' ' + cls : '');
}

// ---- Record ----
$btnRecord.addEventListener('click', async function() {
  $btnRecord.disabled = true;

  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
  } catch (e) {
    setStatus('Cancelled', '');
    $btnRecord.disabled = false;
    return;
  }

  $preview.srcObject = stream;
  $hint.textContent = '';

  try {
    var title = 'Recording ' + new Date().toLocaleString();
    var job = await api('POST', '/jobs', { title: title });
    jobId = job.id;
  } catch (err) {
    setStatus('Server error: ' + err.message, 'err');
    stream.getTracks().forEach(function(t) { t.stop(); });
    $btnRecord.disabled = false;
    return;
  }

  // Prefer MP4 (better OpenCV compatibility), fallback to WebM
  var mimeType;
  if (MediaRecorder.isTypeSupported('video/mp4;codecs=avc1')) {
    mimeType = 'video/mp4;codecs=avc1';
  } else if (MediaRecorder.isTypeSupported('video/mp4')) {
    mimeType = 'video/mp4';
  } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
    mimeType = 'video/webm;codecs=vp9';
  } else {
    mimeType = 'video/webm;codecs=vp8';
  }

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType, videoBitsPerSecond: 2500000 });

  mediaRecorder.ondataavailable = function(e) {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  stream.getVideoTracks()[0].onended = function() {
    if (mediaRecorder.state === 'recording') $btnStop.click();
  };

  mediaRecorder.start(1000);
  startTime = Date.now();
  timerInterval = setInterval(tickTimer, 1000);

  $waveform.classList.remove('idle');
  showBtn('stop');
  setStatus('Recording', 'rec');
});

// ---- Stop ----
$btnStop.addEventListener('click', async function() {
  $btnStop.disabled = true;
  clearInterval(timerInterval);
  $waveform.classList.add('idle');
  setStatus('Saving...', 'upload');

  await new Promise(function(resolve) {
    mediaRecorder.onstop = resolve;
    mediaRecorder.stop();
  });

  stream.getTracks().forEach(function(t) { t.stop(); });
  $preview.srcObject = null;

  var blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
  recordedChunks = [];

  $preview.src = URL.createObjectURL(blob);
  $preview.muted = true;
  $preview.controls = true;
  $hint.textContent = 'Review your recording, then click the green button to process';

  setStatus('Uploading...', 'upload');
  try {
    await uploadBlob(jobId, blob);
    setStatus('Ready to process', 'done');
    showBtn('process');
  } catch (err) {
    setStatus('Upload failed: ' + err.message, 'err');
    showBtn('record');
    $btnRecord.disabled = false;
  }
});

// ---- Process (extract frames) ----
$btnProcess.addEventListener('click', async function() {
  $btnProcess.disabled = true;
  showBtn('spinner');
  setStatus('Extracting frames...', 'upload');

  try {
    await api('POST', '/jobs/' + jobId + '/extract-frames');
    setStatus('Splitting video into frames...', 'done');
    $hint.textContent = 'Extracting frames from your recording. This may take a moment...';
    // Poll until extraction completes, then redirect to frame review
    var pollInterval = setInterval(async function() {
      try {
        var status = await api('GET', '/jobs/' + jobId + '/status');
        if (status.status === 'completed') {
          clearInterval(pollInterval);
          window.location.href = '/jobs/' + jobId + '/frames';
        } else if (status.status === 'failed') {
          clearInterval(pollInterval);
          setStatus('Frame extraction failed', 'err');
          showBtn('record');
          $btnRecord.disabled = false;
        }
      } catch(e) {}
    }, 2000);
  } catch (err) {
    setStatus('Error: ' + err.message, 'err');
    showBtn('process');
    $btnProcess.disabled = false;
  }
});

$dashboard.addEventListener('click', function() { window.location.href = '/#/'; });
