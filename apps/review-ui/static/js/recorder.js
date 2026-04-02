var API_BASE = '/api';

var $btnRecord    = document.getElementById('btn-record');
var $btnStop      = document.getElementById('btn-stop');
var $btnProcess   = document.getElementById('btn-process');
var $btnRerecord  = document.getElementById('btn-rerecord');
var $pillExtract  = document.getElementById('pill-extract');
var $pillRerecord = document.getElementById('pill-rerecord');
var $spinner      = document.getElementById('spinner');
var $stopBanner   = document.getElementById('stop-banner');
var $waveform   = document.getElementById('waveform');
var $timer      = document.getElementById('timer');
var $status     = document.getElementById('status');
var $preview    = document.getElementById('preview');
var $hint       = document.getElementById('hint');
var $dashboard  = document.getElementById('btn-dashboard');

// Crop overlay elements
var $cropOverlay   = document.getElementById('crop-overlay');
var $cropVideo     = document.getElementById('crop-video');
var $cropCanvas    = document.getElementById('crop-canvas');
var $cropConfirm   = document.getElementById('crop-confirm');
var $cropFull      = document.getElementById('crop-full');
var $cropCancel    = document.getElementById('crop-cancel');
var $cropSizeLabel = document.getElementById('crop-size-label');

var mediaRecorder = null;
var recordedChunks = [];
var stream = null;
var startTime = 0;
var timerInterval = null;
var jobId = null;

// Crop region in actual video pixel coordinates (null = full screen)
var cropRegion = null; // {x, y, w, h}

// ---- API helpers ----

function api(method, path, body) {
  var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
  if (body && method !== 'GET') opts.body = JSON.stringify(body);
  return fetch(API_BASE + path, opts).then(function(res) {
    if (!res.ok) return res.json().catch(function() { return { detail: res.statusText }; }).then(function(err) { throw new Error(err.detail || 'API ' + res.status); });
    return res.json();
  });
}

function uploadBlob(id, blob) {
  var form = new FormData();
  var ext = blob.type.indexOf('mp4') >= 0 ? '.mp4' : '.webm';
  form.append('file', blob, 'recording_' + Date.now() + ext);
  return fetch(API_BASE + '/jobs/' + id + '/upload', { method: 'POST', body: form }).then(function(res) {
    if (!res.ok) throw new Error('Upload failed');
    return res.json();
  });
}

function tickTimer() {
  var s = Math.floor((Date.now() - startTime) / 1000);
  var m = Math.floor(s / 60).toString().padStart(2, '0');
  var sec = (s % 60).toString().padStart(2, '0');
  $timer.textContent = m + ':' + sec;
}

function showBtn(name) {
  $btnRecord.classList.toggle('hidden', name !== 'record');
  $btnStop.classList.toggle('hidden', name !== 'stop');
  // Old icon buttons stay hidden — we use pill buttons now
  $btnProcess.classList.add('hidden');
  $btnRerecord.classList.add('hidden');
  // Show labeled pill buttons after recording stops
  var showReview = (name === 'process');
  $pillExtract.style.display = showReview ? 'inline-flex' : 'none';
  $pillRerecord.style.display = showReview ? 'inline-flex' : 'none';
  $spinner.classList.toggle('hidden', name !== 'spinner');
  // Big floating stop banner
  $stopBanner.style.display = (name === 'stop') ? 'block' : 'none';
}

function setStatus(text, cls) {
  $status.textContent = text;
  $status.className = 'status' + (cls ? ' ' + cls : '');
}

// ===========================================================
//  Crop Region Selection
//  The user drags a rectangle on the live screen preview.
//  We store the coordinates in VIDEO pixels (not screen pixels).
//  Recording always captures the FULL screen for smooth playback.
//  The crop is applied server-side during frame extraction.
// ===========================================================

var cropDrag = { active: false, startX: 0, startY: 0, rect: null };

function showCropOverlay() {
  $cropOverlay.classList.add('active');
  $cropVideo.srcObject = stream;
  $cropCanvas.width = window.innerWidth;
  $cropCanvas.height = window.innerHeight;
  cropDrag.rect = null;
  $cropConfirm.disabled = true;
  $cropSizeLabel.classList.add('hidden');
  drawCropOverlay();
}

function hideCropOverlay() {
  $cropOverlay.classList.remove('active');
  $cropVideo.srcObject = null;
}

function getVideoLayout() {
  // Compute how the video maps to the window (object-fit: contain)
  var track = stream.getVideoTracks()[0];
  var settings = track.getSettings();
  var videoW = settings.width  || 1920;
  var videoH = settings.height || 1080;
  var winW = window.innerWidth;
  var winH = window.innerHeight;
  var videoAspect = videoW / videoH;
  var winAspect   = winW  / winH;
  var dispW, dispH, offsetX, offsetY;

  if (videoAspect > winAspect) {
    dispW = winW;  dispH = winW / videoAspect;
    offsetX = 0;   offsetY = (winH - dispH) / 2;
  } else {
    dispH = winH;  dispW = winH * videoAspect;
    offsetX = (winW - dispW) / 2;  offsetY = 0;
  }
  return { videoW: videoW, videoH: videoH, dispW: dispW, dispH: dispH, offsetX: offsetX, offsetY: offsetY };
}

function screenToVideo(sx, sy) {
  var l = getVideoLayout();
  var vx = (sx - l.offsetX) / l.dispW * l.videoW;
  var vy = (sy - l.offsetY) / l.dispH * l.videoH;
  return {
    x: Math.max(0, Math.min(l.videoW, Math.round(vx))),
    y: Math.max(0, Math.min(l.videoH, Math.round(vy))),
  };
}

function drawCropOverlay() {
  var ctx = $cropCanvas.getContext('2d');
  ctx.clearRect(0, 0, $cropCanvas.width, $cropCanvas.height);
  if (!cropDrag.rect) return;

  var r = cropDrag.rect;

  // Dim everything outside the selection
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(0, 0, $cropCanvas.width, r.y);                          // top
  ctx.fillRect(0, r.y, r.x, r.h);                                     // left
  ctx.fillRect(r.x + r.w, r.y, $cropCanvas.width - r.x - r.w, r.h);  // right
  ctx.fillRect(0, r.y + r.h, $cropCanvas.width, $cropCanvas.height - r.y - r.h); // bottom

  // Selection border
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 3]);
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  ctx.setLineDash([]);

  // Corner handles
  ctx.fillStyle = '#fff';
  var hs = 6;
  [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]].forEach(function(c) {
    ctx.fillRect(c[0] - hs, c[1] - hs, hs * 2, hs * 2);
  });
}

// --- Mouse / touch handlers for crop ---

$cropCanvas.addEventListener('mousedown', function(e) {
  cropDrag.active = true;
  cropDrag.startX = e.clientX;
  cropDrag.startY = e.clientY;
  cropDrag.rect = null;
  $cropConfirm.disabled = true;
  $cropSizeLabel.classList.add('hidden');
});

$cropCanvas.addEventListener('mousemove', function(e) {
  if (!cropDrag.active) return;
  var x = Math.min(cropDrag.startX, e.clientX);
  var y = Math.min(cropDrag.startY, e.clientY);
  var w = Math.abs(e.clientX - cropDrag.startX);
  var h = Math.abs(e.clientY - cropDrag.startY);

  if (w > 10 && h > 10) {
    cropDrag.rect = { x: x, y: y, w: w, h: h };
    $cropConfirm.disabled = false;

    var v1 = screenToVideo(x, y);
    var v2 = screenToVideo(x + w, y + h);
    $cropSizeLabel.textContent = Math.abs(v2.x - v1.x) + ' \u00d7 ' + Math.abs(v2.y - v1.y) + ' px';
    $cropSizeLabel.classList.remove('hidden');
  }
  drawCropOverlay();
});

$cropCanvas.addEventListener('mouseup', function() { cropDrag.active = false; });

// Confirm crop selection
$cropConfirm.addEventListener('click', function() {
  if (!cropDrag.rect) return;
  var r = cropDrag.rect;
  var v1 = screenToVideo(r.x, r.y);
  var v2 = screenToVideo(r.x + r.w, r.y + r.h);

  var cx = Math.min(v1.x, v2.x);
  var cy = Math.min(v1.y, v2.y);
  var cw = Math.abs(v2.x - v1.x);
  var ch = Math.abs(v2.y - v1.y);
  // Even dimensions for codec compatibility
  cw = cw - (cw % 2);
  ch = ch - (ch % 2);

  cropRegion = { x: cx, y: cy, w: cw, h: ch };
  hideCropOverlay();
  beginRecording();
});

// Record full screen (skip crop)
$cropFull.addEventListener('click', function() {
  cropRegion = null;
  hideCropOverlay();
  beginRecording();
});

// Cancel
$cropCancel.addEventListener('click', function() {
  hideCropOverlay();
  if (stream) stream.getTracks().forEach(function(t) { t.stop(); });
  stream = null;
  setStatus('Cancelled', '');
  $btnRecord.disabled = false;
});

// ===========================================================
//  Recording — always records the full screen stream.
//  Crop is stored and applied server-side by OpenCV.
// ===========================================================

$btnRecord.addEventListener('click', function() {
  $btnRecord.disabled = true;

  navigator.mediaDevices.getDisplayMedia({
    video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: false,
  }).then(function(s) {
    stream = s;
    setStatus('Select area to record...', 'upload');
    showCropOverlay();
  }).catch(function() {
    setStatus('Cancelled', '');
    $btnRecord.disabled = false;
  });
});

function beginRecording() {
  var now = new Date();
  var title = 'Recording ' + now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' + now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  api('POST', '/jobs', { title: title }).then(function(job) {
    jobId = job.id;

    // Always record the full screen stream directly — no canvas intermediary
    $preview.srcObject = stream;
    if (cropRegion) {
      $hint.textContent = 'Recording... (region ' + cropRegion.w + '\u00d7' + cropRegion.h + ' will be cropped during extraction)';
    } else {
      $hint.textContent = 'Recording full screen...';
    }

    // Choose codec: prefer MP4, fall back to WebM
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

    // If user stops sharing, auto-stop recording
    stream.getVideoTracks()[0].onended = function() {
      if (mediaRecorder && mediaRecorder.state === 'recording') $btnStop.click();
    };

    mediaRecorder.start(1000);
    startTime = Date.now();
    timerInterval = setInterval(tickTimer, 1000);

    $waveform.classList.remove('idle');
    $btnStop.disabled = false;
    showBtn('stop');
    setStatus('Recording', 'rec');

  }).catch(function(err) {
    setStatus('Server error: ' + err.message, 'err');
    if (stream) stream.getTracks().forEach(function(t) { t.stop(); });
    $btnRecord.disabled = false;
  });
}

// ---- Stop ----
$btnStop.addEventListener('click', function() {
  $btnStop.disabled = true;
  clearInterval(timerInterval);
  $waveform.classList.add('idle');
  setStatus('Saving...', 'upload');

  new Promise(function(resolve) {
    mediaRecorder.onstop = resolve;
    mediaRecorder.stop();
  }).then(function() {
    stream.getTracks().forEach(function(t) { t.stop(); });
    $preview.srcObject = null;

    var blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
    recordedChunks = [];

    $preview.src = URL.createObjectURL(blob);
    $preview.muted = true;
    $preview.controls = true;
    $hint.textContent = 'Review your recording, then click the green button to extract frames';

    setStatus('Uploading...', 'upload');
    return uploadBlob(jobId, blob);
  }).then(function() {
    // If crop region was set, save it to the job so the server can apply it
    if (cropRegion) {
      return api('POST', '/jobs/' + jobId + '/crop', cropRegion).then(function() {
        setStatus('Ready to extract frames (cropped ' + cropRegion.w + '\u00d7' + cropRegion.h + ')', 'done');
      });
    } else {
      setStatus('Ready to extract frames', 'done');
    }
  }).then(function() {
    showBtn('process');
  }).catch(function(err) {
    setStatus('Upload failed: ' + err.message, 'err');
    showBtn('record');
    $btnRecord.disabled = false;
  });
});

// ---- Process (extract frames) ----
$btnProcess.addEventListener('click', function() {
  $btnProcess.disabled = true;
  showBtn('spinner');
  setStatus('Extracting frames...', 'upload');

  api('POST', '/jobs/' + jobId + '/extract-frames').then(function() {
    setStatus('Splitting video into frames...', 'done');
    $hint.textContent = 'Extracting frames from your recording. This may take a moment...';
    var pollInterval = setInterval(function() {
      api('GET', '/jobs/' + jobId + '/status').then(function(st) {
        if (st.status === 'completed') {
          clearInterval(pollInterval);
          window.location.href = '/jobs/' + jobId + '/frames';
        } else if (st.status === 'failed') {
          clearInterval(pollInterval);
          setStatus('Frame extraction failed', 'err');
          showBtn('record');
          $btnRecord.disabled = false;
        }
      }).catch(function() {});
    }, 2000);
  }).catch(function(err) {
    setStatus('Error: ' + err.message, 'err');
    showBtn('process');
    $btnProcess.disabled = false;
  });
});

// ---- Pill button wiring ----
$pillExtract.addEventListener('click', function() { $btnProcess.click(); });
$pillRerecord.addEventListener('click', function() { $btnRerecord.click(); });

// ---- Helper to trigger stop regardless of button state ----
function triggerStop() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    $btnStop.disabled = false;
    $btnStop.click();
  }
}

// ---- Stop banner click ----
$stopBanner.addEventListener('click', triggerStop);

// ---- Escape key to stop recording ----
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    e.preventDefault();
    triggerStop();
  }
});

// ---- Re-record (discard and start over) ----
$btnRerecord.addEventListener('click', function() {
  if (!confirm('Discard this recording and start over?')) return;

  $btnRerecord.disabled = true;
  $btnProcess.disabled = true;
  setStatus('Deleting...', 'upload');

  // Delete the current job from the server
  var deletePromise = jobId
    ? fetch(API_BASE + '/jobs/' + jobId, { method: 'DELETE' }).catch(function() {})
    : Promise.resolve();

  deletePromise.then(function() {
    // Reset all state
    jobId = null;
    cropRegion = null;
    recordedChunks = [];
    mediaRecorder = null;
    stream = null;
    clearInterval(timerInterval);

    // Reset UI
    $preview.srcObject = null;
    $preview.removeAttribute('src');
    $preview.controls = false;
    $timer.textContent = '00:00';
    $hint.textContent = 'Click the red button to start screen recording. After recording, extract frames to review them.';

    showBtn('record');
    $btnRecord.disabled = false;
    $btnRerecord.disabled = false;
    $btnProcess.disabled = false;
    setStatus('Ready', '');
  });
});

$dashboard.addEventListener('click', function() { window.location.href = '/#/'; });
