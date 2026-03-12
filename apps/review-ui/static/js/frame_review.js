/**
 * PolicyCapture - Frame Review Page
 * Shows all sampled frames, highlights auto-selected ones,
 * allows manual selection/deselection.
 * Supports annotated view with bounding boxes for UI element detection.
 */

var API = '/api';
var JOB_ID = window.location.pathname.split('/')[2]; // /jobs/{id}/frames

var allFrames = [];
var screenshots = [];  // auto-selected screenshots
var selectedFrameIds = new Set(); // source_frame_id values that are screenshots
var screenshotMap = {};  // source_frame_id -> screenshot object
var currentFilter = 'all';
var showAnnotated = false; // toggle for bounding box view

// ---- API ----

function api(method, path, body) {
  var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
  if (body && method !== 'GET') opts.body = JSON.stringify(body);
  return fetch(API + path, opts).then(function(res) {
    if (!res.ok) return res.json().then(function(e) { throw new Error(e.detail || res.statusText); });
    return res.json();
  });
}

// ---- Load data ----

function loadData() {
  Promise.all([
    api('GET', '/jobs/' + JOB_ID),
    api('GET', '/jobs/' + JOB_ID + '/frames?limit=2000'),
    api('GET', '/jobs/' + JOB_ID + '/screenshots'),
  ]).then(function(results) {
    var job = results[0];
    allFrames = results[1];
    screenshots = results[2];

    document.getElementById('job-title').textContent = job.title || 'Frame Review';

    // Build lookup of selected frames
    selectedFrameIds.clear();
    screenshotMap = {};
    screenshots.forEach(function(ss) {
      selectedFrameIds.add(ss.source_frame_id);
      screenshotMap[ss.source_frame_id] = ss;
    });

    updateStats();
    renderFrames();
  }).catch(function(err) {
    document.getElementById('frame-grid').innerHTML =
      '<div class="empty-state"><p>Error loading frames</p><p class="sub">' + err.message + '</p></div>';
  });
}

function updateStats() {
  var sceneChanges = allFrames.filter(function(f) { return f.candidate_score > 0.3; });
  document.getElementById('stat-total').innerHTML = 'Total: <b>' + allFrames.length + '</b>';
  document.getElementById('stat-selected').innerHTML = 'Selected: <b>' + selectedFrameIds.size + '</b>';
  document.getElementById('stat-scene').innerHTML = 'Scene changes: <b>' + sceneChanges.length + '</b>';
}

// ---- Render ----

function formatTime(ms) {
  var s = Math.floor(ms / 1000);
  var m = Math.floor(s / 60);
  s = s % 60;
  return m.toString().padStart(2, '0') + ':' + s.toString().padStart(2, '0');
}

function getFrameThumbUrl(frame) {
  var thumbName = 'frame_thumb_' + frame.frame_index.toString().padStart(6, '0') + '.jpg';
  return API + '/artifacts/' + JOB_ID + '/processed_frames/' + thumbName;
}

function getFrameFullUrl(frame) {
  var parts = frame.source_image_path.split('/');
  var filename = parts[parts.length - 1];
  return API + '/artifacts/' + JOB_ID + '/frames/' + filename;
}

function getAnnotatedUrl(frame) {
  // Annotated images are saved as frame_XXXXXX_XXXXms_annotated.jpg alongside the original
  var parts = frame.source_image_path.split('/');
  var filename = parts[parts.length - 1];
  var annotatedName = filename.replace('.png', '_annotated.jpg');
  return API + '/artifacts/' + JOB_ID + '/frames/' + annotatedName;
}

function renderFrames() {
  var grid = document.getElementById('frame-grid');
  var filtered = filterFrames(allFrames);

  if (filtered.length === 0) {
    grid.innerHTML = '<div class="empty-state"><p>No frames match this filter</p></div>';
    return;
  }

  var html = '';
  filtered.forEach(function(frame) {
    var frameRef = 'frame_' + frame.frame_index.toString().padStart(6, '0');
    var isSelected = selectedFrameIds.has(frameRef);
    var isAuto = isSelected && screenshotMap[frameRef] && screenshotMap[frameRef].rationale !== 'Manually selected';
    var isManual = isSelected && screenshotMap[frameRef] && screenshotMap[frameRef].rationale === 'Manually selected';
    var isSceneChange = frame.candidate_score > 0.3;

    var cardClass = 'frame-card';
    if (isAuto) cardClass += ' auto-selected';
    else if (isManual) cardClass += ' manually-selected';
    else if (isSceneChange) cardClass += ' scene-change';

    var badges = '';
    if (isAuto) badges += '<span class="badge badge-auto">Auto</span>';
    if (isManual) badges += '<span class="badge badge-manual">Manual</span>';
    if (isSceneChange && !isSelected) badges += '<span class="badge badge-scene">Scene change</span>';

    var relW = Math.round(frame.relevance_score * 100);
    var sceneW = Math.round(frame.candidate_score * 100);
    var blurW = Math.round(frame.blur_score * 100);

    // Show annotated thumbnail if toggle is on and it's a scene-change frame
    var thumbSrc = getFrameThumbUrl(frame);

    // Show OCR text preview if available
    var textPreview = '';
    if (frame.extracted_text && frame.extracted_text.length > 0) {
      var preview = frame.extracted_text.substring(0, 60);
      if (frame.extracted_text.length > 60) preview += '...';
      textPreview = '<div class="frame-ocr-preview" title="' + frame.extracted_text.replace(/"/g, '&quot;').substring(0, 200) + '">'
        + '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg> '
        + preview + '</div>';
    }

    html += '<div class="' + cardClass + '" data-frame-id="' + frame.id + '" data-frame-index="' + frame.frame_index + '">'
      + '<img src="' + thumbSrc + '" alt="Frame ' + frame.frame_index + '" loading="lazy">'
      + '<div class="frame-badges">' + badges + '</div>'
      + '<div class="frame-check">'
      + '<svg class="check-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>'
      + '</div>'
      + '<div class="frame-scores">'
      + '<div class="score-seg relevance" style="width:' + relW + '%"></div>'
      + '<div class="score-seg scene" style="width:' + sceneW + '%"></div>'
      + '<div class="score-seg blur" style="width:' + blurW + '%"></div>'
      + '</div>'
      + '<div class="frame-info">'
      + '<span class="frame-time">' + formatTime(frame.timestamp_ms) + '</span>'
      + '<span class="frame-score-text">Score: <b>' + frame.candidate_score.toFixed(2) + '</b></span>'
      + '</div>'
      + textPreview
      + '</div>';
  });

  grid.innerHTML = html;

  // Attach click handlers
  grid.querySelectorAll('.frame-card').forEach(function(card) {
    card.addEventListener('click', function(e) {
      // Double-click image: show lightbox
      if (e.target.tagName === 'IMG' && e.detail === 2) {
        showLightbox(card.dataset.frameIndex);
        return;
      }
      // Single click: toggle selection
      toggleFrame(card.dataset.frameId, card.dataset.frameIndex);
    });
  });
}

function filterFrames(frames) {
  if (currentFilter === 'all') return frames;

  return frames.filter(function(f) {
    var ref = 'frame_' + f.frame_index.toString().padStart(6, '0');
    if (currentFilter === 'selected') return selectedFrameIds.has(ref);
    if (currentFilter === 'scene') return f.candidate_score > 0.3;
    if (currentFilter === 'high') return f.candidate_score > 0.2;
    return true;
  });
}

// ---- Toggle selection ----

function toggleFrame(frameId, frameIndex) {
  var ref = 'frame_' + parseInt(frameIndex).toString().padStart(6, '0');

  if (selectedFrameIds.has(ref)) {
    // Deselect: delete the screenshot
    var ss = screenshotMap[ref];
    if (ss) {
      api('DELETE', '/screenshots/' + ss.id).then(function() {
        selectedFrameIds.delete(ref);
        delete screenshotMap[ref];
        updateStats();
        renderFrames();
      });
    }
  } else {
    // Select: promote the frame
    api('POST', '/frames/' + frameId + '/promote').then(function(ss) {
      selectedFrameIds.add(ref);
      screenshotMap[ref] = ss;
      updateStats();
      renderFrames();
    }).catch(function(err) {
      console.error('Promote failed:', err);
    });
  }
}

// ---- Lightbox ----

var lightboxShowAnnotated = false;

function showLightbox(frameIndex) {
  var frame = allFrames.find(function(f) { return f.frame_index === parseInt(frameIndex); });
  if (!frame) return;

  var lightbox = document.getElementById('lightbox');
  var img = document.getElementById('lightbox-img');
  var toggleBtn = document.getElementById('lightbox-annotated-toggle');
  var ocrPanel = document.getElementById('lightbox-ocr');

  lightboxShowAnnotated = false;
  img.src = getFrameFullUrl(frame);
  img.dataset.frameIndex = frameIndex;
  img.dataset.originalSrc = getFrameFullUrl(frame);
  img.dataset.annotatedSrc = getAnnotatedUrl(frame);

  if (toggleBtn) {
    toggleBtn.textContent = 'Show Bounding Boxes';
    toggleBtn.className = 'lightbox-toggle-btn';
  }

  // Show OCR text in lightbox
  if (ocrPanel) {
    if (frame.extracted_text && frame.extracted_text.length > 0) {
      ocrPanel.textContent = frame.extracted_text;
      ocrPanel.style.display = 'block';
    } else {
      ocrPanel.style.display = 'none';
    }
  }

  lightbox.classList.add('open');
}

// Toggle annotated/original in lightbox
document.addEventListener('click', function(e) {
  if (e.target && e.target.id === 'lightbox-annotated-toggle') {
    var img = document.getElementById('lightbox-img');
    lightboxShowAnnotated = !lightboxShowAnnotated;
    if (lightboxShowAnnotated) {
      img.src = img.dataset.annotatedSrc;
      e.target.textContent = 'Show Original';
      e.target.className = 'lightbox-toggle-btn active';
    } else {
      img.src = img.dataset.originalSrc;
      e.target.textContent = 'Show Bounding Boxes';
      e.target.className = 'lightbox-toggle-btn';
    }
  }
});

document.getElementById('lightbox-close').addEventListener('click', function() {
  document.getElementById('lightbox').classList.remove('open');
});

document.getElementById('lightbox').addEventListener('click', function(e) {
  if (e.target === this) this.classList.remove('open');
});

// ---- Filters ----

document.querySelectorAll('.filter-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    document.querySelectorAll('.filter-chip').forEach(function(c) { c.classList.remove('active'); });
    chip.classList.add('active');
    currentFilter = chip.dataset.filter;
    renderFrames();
  });
});

// ---- Done button ----

document.getElementById('btn-done').addEventListener('click', function() {
  window.location.href = '/#/jobs/' + JOB_ID;
});

// ---- Init ----
loadData();
