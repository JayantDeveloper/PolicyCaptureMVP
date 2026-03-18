/**
 * PolicyCapture - Frame Review Page
 */

var API = '/api';
var JOB_ID = window.location.pathname.split('/')[2];

var allFrames = [];
var screenshots = [];
var selectedFrameIds = new Set();
var screenshotMap = {};
var currentFilter = 'all';
var sideTab = 'auto';
var activeFrameIndex = -1;
var showAnnotatedPreview = false;
var videoDurationMs = 0;
var scrubTimestampMs = 0;
var scoreThreshold = 0;

// ---- DOM refs ----
var $sideList = document.getElementById('side-list');
var $sideCount = document.getElementById('side-count');
var $timelineStrip = document.getElementById('timeline-strip');
var $timelineScroll = document.getElementById('timeline-scroll');
var $previewImg = document.getElementById('preview-img');
var $previewEmpty = document.getElementById('preview-empty');
var $previewInfo = document.getElementById('preview-info');
var $previewTime = document.getElementById('preview-time');
var $previewScore = document.getElementById('preview-score');
var $previewBboxToggle = document.getElementById('preview-bbox-toggle');
var $previewSelectBtn = document.getElementById('preview-select-btn');
var $videoTimeline = document.getElementById('video-timeline');
var $vtPlayhead = document.getElementById('vt-playhead');
var $vtHoverTime = document.getElementById('vt-hover-time');
var $vtRuler = document.getElementById('vt-ruler');
var $extractAtBtn = document.getElementById('extract-at-btn');
var $scoreSlider = document.getElementById('score-slider');
var $scoreSliderValue = document.getElementById('score-slider-value');
var $scoreThresholdInfo = document.getElementById('score-threshold-info');

// ---- API (plain fetch, no extra headers when no body) ----
function api(method, path, body) {
  var opts = { method: method };
  if (body && method !== 'GET') {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  return fetch(API + path, opts).then(function(res) {
    if (!res.ok) {
      return res.text().then(function(text) {
        try { var j = JSON.parse(text); throw new Error(j.detail || res.statusText); }
        catch(e) { if (e.message) throw e; throw new Error(text || res.statusText); }
      });
    }
    return res.json();
  });
}

// ---- Per-frame confidence ----
function getFrameConfidence(frame) {
  var ocrConf = frame.ocr_confidence != null ? frame.ocr_confidence : 0;
  if (ocrConf > 1) ocrConf = ocrConf / 100;
  var candScore = frame.candidate_score != null ? frame.candidate_score : 0;
  return ocrConf > 0 ? ocrConf : candScore;
}

// ---- Helpers ----
function formatTime(ms) {
  var s = Math.floor(ms / 1000);
  var m = Math.floor(s / 60);
  s = s % 60;
  return m.toString().padStart(2, '0') + ':' + s.toString().padStart(2, '0');
}

function formatTimeFine(ms) {
  var s = Math.floor(ms / 1000);
  var m = Math.floor(s / 60);
  s = s % 60;
  var tenths = Math.floor((ms % 1000) / 100);
  return m.toString().padStart(2, '0') + ':' + s.toString().padStart(2, '0') + '.' + tenths;
}

function frameRef(frame) {
  return 'frame_' + frame.frame_index.toString().padStart(6, '0');
}

function getFrameThumbUrl(frame) {
  if (frame.frame_index >= 100000) {
    return API + '/artifacts/' + JOB_ID + '/processed_frames/frame_thumb_manual_' + frame.timestamp_ms + 'ms.jpg';
  }
  return API + '/artifacts/' + JOB_ID + '/processed_frames/frame_thumb_' + frame.frame_index.toString().padStart(6, '0') + '.jpg';
}

function getFrameFullUrl(frame) {
  var parts = frame.source_image_path.split('/');
  return API + '/artifacts/' + JOB_ID + '/frames/' + parts[parts.length - 1];
}

function getAnnotatedUrl(frame) {
  var parts = frame.source_image_path.split('/');
  var filename = parts[parts.length - 1];
  return API + '/artifacts/' + JOB_ID + '/frames/' + filename.replace('.png', '_annotated.jpg');
}

function isRedundant(frame) {
  var score = frame.candidate_score != null ? frame.candidate_score : 0;
  return score === 0 && frame.frame_index > 0;
}

function isAutoSelected(ref) {
  return selectedFrameIds.has(ref) && screenshotMap[ref] && screenshotMap[ref].rationale !== 'Manually selected';
}

function isManualSelected(ref) {
  return selectedFrameIds.has(ref) && screenshotMap[ref] && screenshotMap[ref].rationale === 'Manually selected';
}

// ---- Load data ----
function loadData() {
  Promise.all([
    api('GET', '/jobs/' + JOB_ID),
    api('GET', '/jobs/' + JOB_ID + '/frames?limit=2000'),
    api('GET', '/jobs/' + JOB_ID + '/screenshots'),
    api('GET', '/jobs/' + JOB_ID + '/video-info').catch(function() { return null; }),
  ]).then(function(results) {
    var job = results[0];
    allFrames = results[1];
    screenshots = results[2];
    var videoInfo = results[3];

    document.getElementById('job-title').textContent = job.title || 'Frame Review';
    allFrames.sort(function(a, b) { return a.timestamp_ms - b.timestamp_ms; });

    selectedFrameIds.clear();
    screenshotMap = {};
    screenshots.forEach(function(ss) {
      selectedFrameIds.add(ss.source_frame_id);
      screenshotMap[ss.source_frame_id] = ss;
    });

    if (videoInfo && videoInfo.duration_ms) {
      videoDurationMs = videoInfo.duration_ms;
    } else if (allFrames.length > 0) {
      videoDurationMs = allFrames[allFrames.length - 1].timestamp_ms + 500;
    }

    updateStats();
    renderVideoTimeline();
    renderTimeline();
    renderSidePanel();

    if (allFrames.length > 0 && activeFrameIndex < 0) {
      setActiveFrame(allFrames[0].frame_index);
    }
  }).catch(function(err) {
    $timelineStrip.innerHTML = '<div class="empty-state"><p>Error: ' + err.message + '</p></div>';
  });
}

function updateStats() {
  var uniqueCount = allFrames.filter(function(f) { return !isRedundant(f); }).length;
  document.getElementById('stat-total').innerHTML = 'Total: <b>' + allFrames.length + '</b>';
  document.getElementById('stat-selected').innerHTML = 'Selected: <b>' + selectedFrameIds.size + '</b>';
  document.getElementById('stat-redundant').innerHTML = 'Unique: <b>' + uniqueCount + '</b>';
  $sideCount.textContent = selectedFrameIds.size;
}

// ---- Video Timeline ----
function renderVideoTimeline() {
  if (videoDurationMs <= 0) return;
  var oldMarkers = $videoTimeline.querySelectorAll('.vt-marker');
  oldMarkers.forEach(function(m) { m.remove(); });

  allFrames.forEach(function(frame) {
    var ref = frameRef(frame);
    var pct = (frame.timestamp_ms / videoDurationMs) * 100;
    var isAuto = isAutoSelected(ref);
    var isManual = isManualSelected(ref);
    var conf = getFrameConfidence(frame);
    var redundant = isRedundant(frame);

    var cls = 'vt-marker';
    if (isAuto) cls += ' auto';
    else if (isManual) cls += ' manual';
    else if (redundant) cls += ' redundant';
    else if (conf > 0.3) cls += ' scene';
    else return;

    if (frame.frame_index === activeFrameIndex) cls += ' active';

    var marker = document.createElement('div');
    marker.className = cls;
    marker.style.left = pct + '%';
    marker.dataset.frameIndex = frame.frame_index;

    var tooltip = document.createElement('div');
    tooltip.className = 'vt-marker-tooltip';
    var typeLabel = isAuto ? 'Auto' : isManual ? 'Manual' : redundant ? 'Redundant' : 'Scene';
    tooltip.textContent = formatTime(frame.timestamp_ms) + ' · ' + typeLabel + ' · Score: ' + conf.toFixed(2);
    marker.appendChild(tooltip);

    marker.addEventListener('click', function(e) {
      e.stopPropagation();
      setActiveFrame(frame.frame_index);
    });

    $videoTimeline.appendChild(marker);
  });

  renderTimeRuler();
}

function renderTimeRuler() {
  $vtRuler.innerHTML = '';
  if (videoDurationMs <= 0) return;
  var totalSec = videoDurationMs / 1000;
  var tickInterval;
  if (totalSec <= 30) tickInterval = 5;
  else if (totalSec <= 120) tickInterval = 15;
  else if (totalSec <= 300) tickInterval = 30;
  else tickInterval = 60;

  for (var s = 0; s <= totalSec; s += tickInterval) {
    var pct = (s * 1000 / videoDurationMs) * 100;
    var tick = document.createElement('div');
    tick.className = 'vt-tick';
    tick.style.left = pct + '%';
    tick.textContent = formatTime(s * 1000);
    $vtRuler.appendChild(tick);
  }
}

// ---- Video timeline interaction ----
$videoTimeline.addEventListener('mousemove', function(e) {
  var rect = $videoTimeline.getBoundingClientRect();
  var pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  $vtHoverTime.style.display = 'block';
  $vtHoverTime.style.left = (pct * 100) + '%';
  $vtHoverTime.textContent = formatTimeFine(pct * videoDurationMs);
});
$videoTimeline.addEventListener('mouseleave', function() { $vtHoverTime.style.display = 'none'; });
$videoTimeline.addEventListener('click', function(e) {
  if (e.target.closest('.vt-marker')) return;
  var rect = $videoTimeline.getBoundingClientRect();
  var pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  var ms = Math.round(pct * videoDurationMs);
  scrubTimestampMs = ms;
  $vtPlayhead.style.left = (pct * 100) + '%';
  $extractAtBtn.disabled = false;
  $extractAtBtn.textContent = '+ Extract Frame at ' + formatTimeFine(ms);
  var nearest = findNearestFrame(ms);
  if (nearest) setActiveFrame(nearest.frame_index);
});
$videoTimeline.addEventListener('dblclick', function(e) {
  if (e.target.closest('.vt-marker')) return;
  var rect = $videoTimeline.getBoundingClientRect();
  var pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  extractFrameAt(Math.round(pct * videoDurationMs));
});

function findNearestFrame(ms) {
  var best = null, bestDist = Infinity;
  allFrames.forEach(function(f) {
    var d = Math.abs(f.timestamp_ms - ms);
    if (d < bestDist) { bestDist = d; best = f; }
  });
  return best;
}

// ---- Extract frame at timestamp ----
$extractAtBtn.addEventListener('click', function() {
  if (scrubTimestampMs <= 0 && scrubTimestampMs !== 0) return;
  extractFrameAt(scrubTimestampMs);
});

function extractFrameAt(timestampMs) {
  $extractAtBtn.disabled = true;
  $extractAtBtn.textContent = 'Extracting...';
  api('POST', '/jobs/' + JOB_ID + '/extract-frame-at', { timestamp_ms: timestampMs })
    .then(function(newFrame) {
      allFrames.push(newFrame);
      allFrames.sort(function(a, b) { return a.timestamp_ms - b.timestamp_ms; });
      return api('POST', '/frames/' + newFrame.id + '/promote').then(function(ss) {
        var ref = frameRef(newFrame);
        selectedFrameIds.add(ref);
        screenshotMap[ref] = ss;
        updateStats(); renderVideoTimeline(); renderTimeline(); renderSidePanel();
        setActiveFrame(newFrame.frame_index);
        $extractAtBtn.textContent = '+ Extract Frame Here';
        $extractAtBtn.disabled = false;
      });
    })
    .catch(function(err) {
      console.error('Extract frame failed:', err);
      $extractAtBtn.textContent = 'Failed - try again';
      $extractAtBtn.disabled = false;
    });
}

// ---- Filtering ----
function filterFrames(frames) {
  var filtered = frames;
  if (currentFilter !== 'all') {
    filtered = filtered.filter(function(f) {
      var ref = frameRef(f);
      if (currentFilter === 'unique') return !isRedundant(f);
      if (currentFilter === 'selected') return selectedFrameIds.has(ref);
      if (currentFilter === 'scene') return getFrameConfidence(f) > 0.3;
      return true;
    });
  }
  if (scoreThreshold > 0) {
    filtered = filtered.filter(function(f) {
      return getFrameConfidence(f) >= scoreThreshold;
    });
  }
  return filtered;
}

// ---- Frame thumbnail strip ----
function renderTimeline() {
  var filtered = filterFrames(allFrames);
  if (filtered.length === 0) {
    $timelineStrip.innerHTML = '<div class="empty-state"><p>No frames match filter</p></div>';
    return;
  }
  var html = '';
  filtered.forEach(function(frame) {
    var ref = frameRef(frame);
    var isSelected = selectedFrameIds.has(ref);
    var isAuto = isAutoSelected(ref);
    var isManual = isManualSelected(ref);
    var isActive = frame.frame_index === activeFrameIndex;
    var redundant = isRedundant(frame);

    var cls = 'tl-frame';
    if (isActive) cls += ' active';
    if (isSelected) cls += ' selected';
    if (isManual) cls += ' manual-sel';
    if (redundant) cls += ' redundant';

    var dot = '';
    if (isAuto) dot = '<div class="tl-marker-dot auto"></div>';
    else if (isManual) dot = '<div class="tl-marker-dot manual-dot"></div>';
    else if (getFrameConfidence(frame) > 0.3 && !redundant) dot = '<div class="tl-marker-dot scene"></div>';

    html += '<div class="' + cls + '" data-frame-index="' + frame.frame_index + '" data-frame-id="' + frame.id + '">'
      + dot
      + '<img src="' + getFrameThumbUrl(frame) + '" alt="' + formatTime(frame.timestamp_ms) + '" loading="lazy">'
      + '</div>';
  });
  $timelineStrip.innerHTML = html;
  $timelineStrip.querySelectorAll('.tl-frame').forEach(function(el) {
    el.addEventListener('click', function() { setActiveFrame(parseInt(el.dataset.frameIndex)); });
    el.addEventListener('dblclick', function(e) { e.preventDefault(); showLightbox(parseInt(el.dataset.frameIndex)); });
  });
  scrollToActive();
}

function scrollToActive() {
  var activeEl = $timelineStrip.querySelector('.tl-frame.active');
  if (activeEl) activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
}

// ---- Side panel ----
function renderSidePanel() {
  var items;
  if (sideTab === 'auto') {
    items = allFrames.filter(function(f) { return getFrameConfidence(f) > 0 && !isRedundant(f); });
  } else if (sideTab === 'manual') {
    items = allFrames.filter(function(f) { return isManualSelected(frameRef(f)); });
  } else {
    items = allFrames.filter(function(f) { return selectedFrameIds.has(frameRef(f)); });
  }

  var preFilterCount = items.length;
  if (scoreThreshold > 0) {
    items = items.filter(function(f) { return getFrameConfidence(f) >= scoreThreshold; });
  }
  if ($scoreThresholdInfo) {
    $scoreThresholdInfo.textContent = items.length + ' of ' + preFilterCount + ' frames shown';
  }

  if (items.length === 0) {
    var msg = sideTab === 'manual' ? 'No manual selections yet.'
      : sideTab === 'auto' ? 'No auto-detected frames' : 'No frames selected';
    $sideList.innerHTML = '<div class="empty-state"><p>' + msg + '</p></div>';
    return;
  }

  var html = '';
  items.forEach(function(frame) {
    var ref = frameRef(frame);
    var isSelected = selectedFrameIds.has(ref);
    var isAuto = isAutoSelected(ref);
    var isManual = isManualSelected(ref);
    var isActive = frame.frame_index === activeFrameIndex;
    var selectedClass = isSelected ? '' : ' unselected';
    var typeLabel = isAuto ? 'Auto' : isManual ? 'Manual' : 'Detected';
    var typeClass = isManual ? ' manual' : '';
    var actionBtn = isSelected
      ? '<button class="side-card-remove" title="Deselect">&times;</button>'
      : '<button class="side-card-add" title="Select">+</button>';

    html += '<div class="side-card' + (isActive ? ' active' : '') + selectedClass
      + '" data-frame-index="' + frame.frame_index + '" data-frame-id="' + frame.id + '">'
      + '<img src="' + getFrameThumbUrl(frame) + '" alt="">'
      + '<div class="side-card-info">'
      + '<span class="side-card-time">' + formatTime(frame.timestamp_ms) + '</span>'
      + '<span class="side-card-score">Score: <b>' + getFrameConfidence(frame).toFixed(2) + '</b></span>'
      + '<span class="side-card-type' + typeClass + '">' + typeLabel + (isSelected ? '' : ' (not selected)') + '</span>'
      + '</div>' + actionBtn + '</div>';
  });
  $sideList.innerHTML = html;

  $sideList.querySelectorAll('.side-card').forEach(function(card) {
    card.addEventListener('click', function(e) {
      if (e.target.closest('.side-card-remove') || e.target.closest('.side-card-add')) return;
      setActiveFrame(parseInt(card.dataset.frameIndex));
    });
    var removeBtn = card.querySelector('.side-card-remove');
    if (removeBtn) removeBtn.addEventListener('click', function() { toggleFrame(card.dataset.frameId, card.dataset.frameIndex); });
    var addBtn = card.querySelector('.side-card-add');
    if (addBtn) addBtn.addEventListener('click', function() { toggleFrame(card.dataset.frameId, card.dataset.frameIndex); });
  });
}

// ---- Side tabs ----
document.querySelectorAll('.side-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.side-tab').forEach(function(t) { t.classList.remove('active'); });
    tab.classList.add('active');
    sideTab = tab.dataset.tab;
    renderSidePanel();
  });
});

// ---- Preview ----
function setActiveFrame(frameIndex) {
  activeFrameIndex = frameIndex;
  showAnnotatedPreview = false;
  var frame = allFrames.find(function(f) { return f.frame_index === frameIndex; });
  if (!frame) return;

  $previewImg.src = getFrameFullUrl(frame);
  $previewImg.style.display = 'block';
  $previewEmpty.style.display = 'none';
  $previewInfo.style.display = 'flex';
  $previewImg.dataset.originalSrc = getFrameFullUrl(frame);
  $previewImg.dataset.annotatedSrc = getAnnotatedUrl(frame);
  $previewTime.textContent = formatTime(frame.timestamp_ms);
  $previewScore.textContent = getFrameConfidence(frame).toFixed(2);
  $previewBboxToggle.className = 'preview-toggle-btn';
  $previewBboxToggle.textContent = 'Bounding Boxes';
  updateSelectButton(frame);

  if (videoDurationMs > 0) {
    $vtPlayhead.style.left = ((frame.timestamp_ms / videoDurationMs) * 100) + '%';
    scrubTimestampMs = frame.timestamp_ms;
  }

  $timelineStrip.querySelectorAll('.tl-frame').forEach(function(el) {
    el.classList.toggle('active', parseInt(el.dataset.frameIndex) === frameIndex);
  });
  $sideList.querySelectorAll('.side-card').forEach(function(el) {
    el.classList.toggle('active', parseInt(el.dataset.frameIndex) === frameIndex);
  });
  $videoTimeline.querySelectorAll('.vt-marker').forEach(function(m) {
    m.classList.toggle('active', parseInt(m.dataset.frameIndex) === frameIndex);
  });
}

function updateSelectButton(frame) {
  var ref = frameRef(frame);
  if (selectedFrameIds.has(ref)) {
    $previewSelectBtn.textContent = 'Deselect';
    $previewSelectBtn.className = 'preview-select-btn deselect';
  } else {
    $previewSelectBtn.textContent = 'Select';
    $previewSelectBtn.className = 'preview-select-btn select';
  }
}

// ---- Toggle selection ----
function toggleFrame(frameId, frameIndex) {
  var ref = 'frame_' + parseInt(frameIndex).toString().padStart(6, '0');
  if (selectedFrameIds.has(ref)) {
    var ss = screenshotMap[ref];
    if (ss) {
      api('DELETE', '/screenshots/' + ss.id).then(function() {
        selectedFrameIds.delete(ref);
        delete screenshotMap[ref];
        afterSelectionChange(frameIndex);
      });
    }
  } else {
    api('POST', '/frames/' + frameId + '/promote').then(function(ss) {
      selectedFrameIds.add(ref);
      screenshotMap[ref] = ss;
      afterSelectionChange(frameIndex);
    }).catch(function(err) { console.error('Promote failed:', err); });
  }
}

function afterSelectionChange(frameIndex) {
  updateStats(); renderVideoTimeline(); renderTimeline(); renderSidePanel();
  var frame = allFrames.find(function(f) { return f.frame_index === parseInt(frameIndex); });
  if (frame) updateSelectButton(frame);
}

// ---- Preview buttons ----
$previewSelectBtn.addEventListener('click', function() {
  if (activeFrameIndex < 0) return;
  var frame = allFrames.find(function(f) { return f.frame_index === activeFrameIndex; });
  if (frame) toggleFrame(frame.id, frame.frame_index);
});

$previewBboxToggle.addEventListener('click', function() {
  showAnnotatedPreview = !showAnnotatedPreview;
  if (showAnnotatedPreview) {
    $previewImg.src = $previewImg.dataset.annotatedSrc;
    $previewBboxToggle.textContent = 'Original';
    $previewBboxToggle.className = 'preview-toggle-btn active';
  } else {
    $previewImg.src = $previewImg.dataset.originalSrc;
    $previewBboxToggle.textContent = 'Bounding Boxes';
    $previewBboxToggle.className = 'preview-toggle-btn';
  }
});

// ---- Keyboard nav ----
document.addEventListener('keydown', function(e) {
  if (document.getElementById('lightbox').classList.contains('open')) {
    if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('open');
    return;
  }
  if (activeFrameIndex < 0) return;
  var filtered = filterFrames(allFrames);
  var curIdx = filtered.findIndex(function(f) { return f.frame_index === activeFrameIndex; });
  if (curIdx < 0) return;
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    e.preventDefault();
    if (curIdx < filtered.length - 1) setActiveFrame(filtered[curIdx + 1].frame_index);
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (curIdx > 0) setActiveFrame(filtered[curIdx - 1].frame_index);
  } else if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault();
    toggleFrame(filtered[curIdx].id, filtered[curIdx].frame_index);
  }
});

// ---- Lightbox ----
var lightboxShowAnnotated = false;
function showLightbox(frameIndex) {
  var frame = allFrames.find(function(f) { return f.frame_index === frameIndex; });
  if (!frame) return;
  var lightbox = document.getElementById('lightbox');
  var img = document.getElementById('lightbox-img');
  var toggleBtn = document.getElementById('lightbox-annotated-toggle');
  var ocrPanel = document.getElementById('lightbox-ocr');
  lightboxShowAnnotated = false;
  img.src = getFrameFullUrl(frame);
  img.dataset.originalSrc = getFrameFullUrl(frame);
  img.dataset.annotatedSrc = getAnnotatedUrl(frame);
  toggleBtn.textContent = 'Show Bounding Boxes';
  toggleBtn.className = 'lightbox-toggle-btn';
  if (frame.extracted_text && frame.extracted_text.length > 0) {
    ocrPanel.textContent = frame.extracted_text;
    ocrPanel.style.display = 'block';
  } else {
    ocrPanel.style.display = 'none';
  }
  lightbox.classList.add('open');
}

document.addEventListener('click', function(e) {
  if (e.target && e.target.id === 'lightbox-annotated-toggle') {
    var img = document.getElementById('lightbox-img');
    lightboxShowAnnotated = !lightboxShowAnnotated;
    img.src = lightboxShowAnnotated ? img.dataset.annotatedSrc : img.dataset.originalSrc;
    e.target.textContent = lightboxShowAnnotated ? 'Show Original' : 'Show Bounding Boxes';
    e.target.className = 'lightbox-toggle-btn' + (lightboxShowAnnotated ? ' active' : '');
  }
});

document.getElementById('lightbox-close').addEventListener('click', function() {
  document.getElementById('lightbox').classList.remove('open');
});
document.getElementById('lightbox').addEventListener('click', function(e) {
  if (e.target === this) this.classList.remove('open');
});

// ---- Filter chips ----
document.querySelectorAll('.filter-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    document.querySelectorAll('.filter-chip').forEach(function(c) { c.classList.remove('active'); });
    chip.classList.add('active');
    currentFilter = chip.dataset.filter;
    renderTimeline();
  });
});

// ---- Done button ----
document.getElementById('btn-done').addEventListener('click', function() {
  window.location.href = '/jobs/' + JOB_ID + '/ocr-review';
});

// ---- Backfill confidence ----
document.getElementById('btn-backfill').addEventListener('click', function() {
  var btn = this;
  btn.disabled = true;
  btn.textContent = 'Computing...';
  api('POST', '/jobs/' + JOB_ID + '/backfill-confidence')
    .then(function(r) {
      btn.disabled = false;
      if (r.updated > 0) { btn.textContent = r.updated + ' updated'; loadData(); }
      else { btn.textContent = 'Up to date'; }
      setTimeout(function() { btn.textContent = 'Refresh Scores'; }, 2000);
    })
    .catch(function(err) { btn.textContent = 'Failed'; btn.disabled = false; });
});

// ==============================================================
//  Select All / Unselect All
// ==============================================================
document.getElementById('btn-select-all').addEventListener('click', function() {
  var btn = this;
  btn.disabled = true;
  btn.textContent = 'Selecting...';
  fetch(API + '/jobs/' + JOB_ID + '/select-all', { method: 'POST' })
    .then(function(res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(function(data) {
      btn.textContent = 'Select All';
      btn.disabled = false;
      loadData();
    })
    .catch(function(err) {
      btn.textContent = 'Select All';
      btn.disabled = false;
      alert('Select all failed: ' + err.message);
    });
});

document.getElementById('btn-unselect-all').addEventListener('click', function() {
  var btn = this;
  btn.disabled = true;
  btn.textContent = 'Clearing...';
  fetch(API + '/jobs/' + JOB_ID + '/unselect-all', { method: 'POST' })
    .then(function(res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(function(data) {
      btn.textContent = 'Unselect All';
      btn.disabled = false;
      loadData();
    })
    .catch(function(err) {
      btn.textContent = 'Unselect All';
      btn.disabled = false;
      alert('Unselect failed: ' + err.message);
    });
});

// ==============================================================
//  Score threshold slider — native <input type="range">
//  No custom styling. Just works.
// ==============================================================
if ($scoreSlider) {
  $scoreSlider.oninput = function() {
    scoreThreshold = Number(this.value) / 100;
    $scoreSliderValue.textContent = scoreThreshold.toFixed(2);
    renderSidePanel();
    renderTimeline();
  };
  $scoreSlider.onchange = function() {
    scoreThreshold = Number(this.value) / 100;
    $scoreSliderValue.textContent = scoreThreshold.toFixed(2);
    renderSidePanel();
    renderTimeline();
  };
}

// ---- Init ----
loadData();
