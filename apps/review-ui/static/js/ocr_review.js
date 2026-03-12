/**
 * PolicyCapture - OCR & Entity Review Page
 * Displays extracted text with entity highlighting, table rendering,
 * search, and annotations.
 */

var API = '/api';
var JOB_ID = window.location.pathname.split('/')[2];

var ocrData = [];
var activeIndex = -1;
var currentTab = 'text';
var activeEntityFilter = null;
var searchQuery = '';
var searchResults = [];
var searchDebounce = null;

// Entity type config
var ENTITY_TYPES = {
  date:          { label: 'Dates',         color: '#f59e0b' },
  currency:      { label: 'Currency',      color: '#22c55e' },
  email:         { label: 'Emails',        color: '#3b82f6' },
  phone:         { label: 'Phone',         color: '#a78bfa' },
  percentage:    { label: 'Percentages',   color: '#14b8a6' },
  person_name:   { label: 'People',        color: '#f472b6' },
  organization:  { label: 'Organizations', color: '#fb923c' },
  address:       { label: 'Addresses',     color: '#818cf8' },
  policy_number: { label: 'Policy #',      color: '#e879f9' },
  case_number:   { label: 'Case #',        color: '#67e8f9' },
  ssn:           { label: 'SSN',           color: '#ef4444' },
  url:           { label: 'URLs',          color: '#94a3b8' },
  id_number:     { label: 'IDs',           color: '#d4d4d8' },
};

// DOM
var $frameList = document.getElementById('frame-list');
var $contentImg = document.getElementById('content-img');
var $contentEmpty = document.getElementById('content-empty');
var $imageInfo = document.getElementById('image-info');
var $imageInfoText = document.getElementById('image-info-text');
var $ocrConfDot = document.getElementById('ocr-conf-dot');
var $textView = document.getElementById('text-view');
var $entitiesView = document.getElementById('entities-view');
var $tablesView = document.getElementById('tables-view');
var $notesView = document.getElementById('notes-view');
var $notesTextarea = document.getElementById('notes-textarea');
var $searchInput = document.getElementById('search-input');
var $searchCount = document.getElementById('search-count');
var $searchResults = document.getElementById('search-results');
var $entityFilters = document.getElementById('entity-filters');
var $processingOverlay = document.getElementById('processing-overlay');
var $processingProgress = document.getElementById('processing-progress');

// ---- API ----
function api(method, path, body) {
  var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
  if (body && method !== 'GET') opts.body = JSON.stringify(body);
  return fetch(API + path, opts).then(function(res) {
    if (!res.ok) return res.json().then(function(e) { throw new Error(e.detail || res.statusText); });
    return res.json();
  });
}

function formatTime(ms) {
  var s = Math.floor(ms / 1000);
  var m = Math.floor(s / 60);
  s = s % 60;
  return m.toString().padStart(2, '0') + ':' + s.toString().padStart(2, '0');
}

// ---- Load ----
function loadData() {
  Promise.all([
    api('GET', '/jobs/' + JOB_ID),
    api('GET', '/jobs/' + JOB_ID + '/ocr-data'),
  ]).then(function(results) {
    var job = results[0];
    ocrData = results[1];
    document.getElementById('job-title').textContent = (job.title || 'OCR Review');
    updateStats();
    renderFrameList();
    renderEntityFilters();
    if (ocrData.length > 0) setActiveItem(0);
  }).catch(function(err) {
    $frameList.innerHTML = '<div class="empty-state"><p>Error: ' + err.message + '</p></div>';
  });
}

function updateStats() {
  document.getElementById('stat-frames').innerHTML = 'Frames: <b>' + ocrData.length + '</b>';
  document.getElementById('frame-count').textContent = ocrData.length;
  var totalEntities = 0;
  var totalTables = 0;
  ocrData.forEach(function(item) {
    if (item.entities && item.entities.summary) {
      totalEntities += item.entities.summary.total_entities || 0;
    }
    if (item.entities && item.entities.tables) {
      totalTables += item.entities.tables.length;
    }
  });
  document.getElementById('stat-entities').innerHTML = 'Entities: <b>' + totalEntities + '</b>';
  document.getElementById('stat-tables').innerHTML = 'Tables: <b>' + totalTables + '</b>';
}

// ---- Frame list ----
function renderFrameList() {
  if (ocrData.length === 0) {
    $frameList.innerHTML = '<div class="empty-state"><p>No frames processed</p></div>';
    return;
  }

  var html = '';
  ocrData.forEach(function(item, idx) {
    var hasText = item.extracted_text && item.extracted_text.length > 0;
    var entityDots = '';

    if (item.entities && item.entities.categories) {
      var cats = item.entities.categories;
      Object.keys(cats).forEach(function(type) {
        if (cats[type] && cats[type].length > 0 && ENTITY_TYPES[type]) {
          entityDots += '<div class="fs-entity-dot" style="background:' + ENTITY_TYPES[type].color + '"></div>';
        }
      });
    }

    var thumbUrl = item.thumbnail_path
      ? API + '/artifacts/' + JOB_ID + '/thumbnails/' + item.thumbnail_path.split('/').pop()
      : API + '/artifacts/' + JOB_ID + '/screenshots/' + item.image_path.split('/').pop();

    var entCount = item.entities && item.entities.summary ? item.entities.summary.total_entities || 0 : 0;
    var tableCount = item.entities && item.entities.tables ? item.entities.tables.length : 0;
    var conf = item.ocr_confidence || 0;
    var confClass = conf >= 70 ? 'conf-high' : conf >= 40 ? 'conf-med' : 'conf-low';

    var metaParts = [];
    if (hasText) metaParts.push(item.extracted_text.length + ' chars');
    else metaParts.push('No OCR');
    if (entCount > 0) metaParts.push(entCount + ' entities');
    if (tableCount > 0) metaParts.push(tableCount + ' table' + (tableCount > 1 ? 's' : ''));

    html += '<div class="fs-card' + (idx === activeIndex ? ' active' : '') + (hasText ? ' has-text' : ' no-text') + '" data-index="' + idx + '">'
      + '<img src="' + thumbUrl + '" alt="" loading="lazy">'
      + '<div class="fs-card-info">'
      + '<span class="fs-card-time">' + formatTime(item.timestamp_ms) + '</span>'
      + '<span class="fs-card-meta">' + metaParts.join(' &middot; ') + '</span>'
      + (entityDots ? '<div class="fs-card-entities">' + entityDots + '</div>' : '')
      + '</div>'
      + (hasText && conf > 0 ? '<span class="conf-badge ' + confClass + '">' + Math.round(conf) + '%</span>' : '')
      + '</div>';
  });

  $frameList.innerHTML = html;

  $frameList.querySelectorAll('.fs-card').forEach(function(card) {
    card.addEventListener('click', function() {
      setActiveItem(parseInt(card.dataset.index));
    });
  });
}

// ---- Entity filter chips ----
function renderEntityFilters() {
  var typeCounts = {};
  ocrData.forEach(function(item) {
    if (item.entities && item.entities.categories) {
      Object.keys(item.entities.categories).forEach(function(type) {
        var vals = item.entities.categories[type];
        if (vals && vals.length > 0) {
          typeCounts[type] = (typeCounts[type] || 0) + vals.length;
        }
      });
    }
  });

  var html = '';
  Object.keys(ENTITY_TYPES).forEach(function(type) {
    if (typeCounts[type]) {
      html += '<button class="entity-chip ent-' + type + ' active" data-type="' + type + '">'
        + ENTITY_TYPES[type].label + ' (' + typeCounts[type] + ')</button>';
    }
  });
  $entityFilters.innerHTML = html;

  $entityFilters.querySelectorAll('.entity-chip').forEach(function(chip) {
    chip.addEventListener('click', function() {
      if (activeEntityFilter === chip.dataset.type) {
        activeEntityFilter = null;
        $entityFilters.querySelectorAll('.entity-chip').forEach(function(c) { c.classList.add('active'); });
      } else {
        activeEntityFilter = chip.dataset.type;
        $entityFilters.querySelectorAll('.entity-chip').forEach(function(c) {
          c.classList.toggle('active', c.dataset.type === activeEntityFilter);
        });
      }
      if (activeIndex >= 0) renderTextView(ocrData[activeIndex]);
    });
  });
}

// ---- Set active item ----
function setActiveItem(idx) {
  activeIndex = idx;
  var item = ocrData[idx];
  if (!item) return;

  // Update image
  var imgUrl = API + '/artifacts/' + JOB_ID + '/screenshots/' + item.image_path.split('/').pop();
  $contentImg.src = imgUrl;
  $contentImg.style.display = 'block';
  $contentEmpty.style.display = 'none';
  $imageInfo.style.display = 'flex';

  var conf = item.ocr_confidence || 0;
  var confColor = conf >= 70 ? '#4ade80' : conf >= 40 ? '#fbbf24' : '#fca5a5';
  $ocrConfDot.style.background = confColor;
  $imageInfoText.textContent = formatTime(item.timestamp_ms)
    + ' \u00b7 ' + (item.rationale || 'Selected')
    + (conf > 0 ? ' \u00b7 OCR: ' + Math.round(conf) + '%' : '');

  // Render all tabs
  renderTextView(item);
  renderEntitiesView(item);
  renderTablesView(item);
  renderNotesView(item);

  // Update frame list active state
  $frameList.querySelectorAll('.fs-card').forEach(function(c, i) {
    c.classList.toggle('active', i === idx);
  });

  // Scroll active card into view
  var activeCard = $frameList.querySelector('.fs-card.active');
  if (activeCard) activeCard.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ---- Text view with entity highlighting ----
function renderTextView(item) {
  if (!item.extracted_text) {
    $textView.innerHTML = '<div class="empty-state"><p>No text extracted</p><p style="font-size:11px;color:#475569">Click "Run OCR & Extract" to process</p></div>';
    return;
  }

  var text = item.extracted_text;
  var entities = (item.entities && item.entities.entities) ? item.entities.entities : [];

  // Filter entities if a type filter is active
  if (activeEntityFilter) {
    entities = entities.filter(function(e) { return e.type === activeEntityFilter; });
  }

  // Build highlighted HTML preserving structure
  var highlighted = buildHighlightedText(text, entities);

  // Apply search highlighting on top
  if (searchQuery) {
    highlighted = highlightSearch(highlighted, searchQuery);
  }

  $textView.innerHTML = '<div class="ocr-text-block">' + highlighted + '</div>';
}

function buildHighlightedText(text, entities) {
  if (!entities || entities.length === 0) {
    return formatStructuredText(escapeHtml(text));
  }

  // Sort ascending by start position
  var sorted = entities.slice().sort(function(a, b) { return a.start - b.start; });
  var parts = [];
  var lastEnd = 0;

  sorted.forEach(function(ent) {
    if (ent.start >= lastEnd && ent.start < text.length) {
      parts.push(escapeHtml(text.substring(lastEnd, ent.start)));
      var entText = text.substring(ent.start, Math.min(ent.end, text.length));
      var cfg = ENTITY_TYPES[ent.type];
      var label = cfg ? cfg.label : ent.type;
      parts.push('<span class="entity-highlight ' + ent.type + '" title="' + label + ': ' + escapeHtml(ent.text) + '">'
        + escapeHtml(entText) + '</span>');
      lastEnd = ent.end;
    }
  });
  parts.push(escapeHtml(text.substring(lastEnd)));

  return formatStructuredText(parts.join(''));
}

function formatStructuredText(html) {
  // Convert line breaks to <br> but collapse triple+ breaks to double
  html = html.replace(/\n{3,}/g, '\n\n');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function highlightSearch(html, query) {
  if (!query) return html;
  var regex = new RegExp('(?<=>)([^<]*?)(' + escapeRegex(query) + ')(?=[^>]*<|[^<]*$)', 'gi');
  return html.replace(regex, function(match, before, found) {
    return before + '<span class="search-highlight">' + found + '</span>';
  });
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ---- Entities view ----
function renderEntitiesView(item) {
  if (!item.entities || !item.entities.categories) {
    $entitiesView.innerHTML = '<div class="empty-state"><p>No entities extracted</p></div>';
    return;
  }

  var cats = item.entities.categories;
  var html = '';

  Object.keys(ENTITY_TYPES).forEach(function(type) {
    var values = cats[type];
    if (!values || values.length === 0) return;

    // Deduplicate values
    var unique = [];
    var seen = {};
    values.forEach(function(v) {
      if (!seen[v]) { seen[v] = true; unique.push(v); }
    });

    var cfg = ENTITY_TYPES[type];
    html += '<div class="entity-group">'
      + '<div class="entity-group-header ent-' + type + '">'
      + '<span>' + cfg.label + '</span>'
      + '<span class="entity-group-count">' + unique.length + '</span>'
      + '</div>'
      + '<div class="entity-group-items">';

    unique.forEach(function(val) {
      html += '<span class="entity-tag ent-' + type + '" title="Click to copy">' + escapeHtml(val) + '</span>';
    });

    html += '</div></div>';
  });

  if (!html) {
    html = '<div class="empty-state"><p>No entities found</p></div>';
  }

  $entitiesView.innerHTML = html;

  // Click to copy
  $entitiesView.querySelectorAll('.entity-tag').forEach(function(tag) {
    tag.addEventListener('click', function() {
      navigator.clipboard.writeText(tag.textContent).then(function() {
        tag.style.transform = 'scale(0.95)';
        tag.style.opacity = '0.7';
        setTimeout(function() { tag.style.transform = ''; tag.style.opacity = ''; }, 200);
      });
    });
  });
}

// ---- Tables view ----
function renderTablesView(item) {
  var tables = (item.entities && item.entities.tables) ? item.entities.tables : [];

  if (tables.length === 0) {
    $tablesView.innerHTML = '<div class="empty-state"><p>No tables detected in this frame</p><p style="font-size:11px;color:#475569">Tables are automatically detected from grid structures in the image</p></div>';
    return;
  }

  var html = '';
  tables.forEach(function(table, ti) {
    html += '<div class="table-card">'
      + '<div class="table-card-header">'
      + '<span class="table-card-title">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>'
      + 'Table ' + (ti + 1)
      + '</span>'
      + '<span class="table-card-meta">' + table.num_rows + ' rows &times; ' + table.num_cols + ' cols</span>'
      + '</div>';

    if (table.rows && table.rows.length > 0) {
      html += '<table class="extracted-table">';

      // First row as header
      html += '<thead><tr>';
      table.rows[0].forEach(function(cell) {
        html += '<th>' + escapeHtml(cell || '') + '</th>';
      });
      html += '</tr></thead>';

      // Remaining rows as body
      if (table.rows.length > 1) {
        html += '<tbody>';
        for (var r = 1; r < table.rows.length; r++) {
          html += '<tr>';
          table.rows[r].forEach(function(cell) {
            html += '<td>' + escapeHtml(cell || '') + '</td>';
          });
          html += '</tr>';
        }
        html += '</tbody>';
      }

      html += '</table>';
    } else if (table.raw_text) {
      html += '<div style="padding:12px;font-family:monospace;font-size:12px;white-space:pre-wrap;color:#94a3b8">'
        + escapeHtml(table.raw_text) + '</div>';
    }

    html += '</div>';
  });

  $tablesView.innerHTML = html;
}

// ---- Notes view ----
function renderNotesView(item) {
  $notesTextarea.value = item.notes || '';
}

// ---- Tabs ----
document.querySelectorAll('.text-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.text-tab').forEach(function(t) { t.classList.remove('active'); });
    tab.classList.add('active');
    currentTab = tab.dataset.tab;

    $textView.style.display = currentTab === 'text' ? 'block' : 'none';
    $entitiesView.style.display = currentTab === 'entities' ? 'flex' : 'none';
    $tablesView.style.display = currentTab === 'tables' ? 'flex' : 'none';
    $notesView.style.display = currentTab === 'notes' ? 'flex' : 'none';

    // Update title
    var titles = { text: 'Extracted Text', entities: 'Named Entities', tables: 'Detected Tables', notes: 'Annotations' };
    document.getElementById('text-panel-title').textContent = titles[currentTab] || 'Extracted Text';
  });
});

// ---- Save notes ----
document.getElementById('btn-save-notes').addEventListener('click', function() {
  if (activeIndex < 0) return;
  var item = ocrData[activeIndex];
  var notes = $notesTextarea.value;

  fetch(API + '/screenshots/' + item.screenshot_id + '/notes', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes: notes }),
  }).then(function(res) {
    if (res.ok) {
      item.notes = notes;
      var $saved = document.getElementById('notes-saved');
      $saved.classList.add('show');
      setTimeout(function() { $saved.classList.remove('show'); }, 2000);
    }
  });
});

// ---- Search ----
$searchInput.addEventListener('input', function() {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(function() {
    searchQuery = $searchInput.value.trim();
    if (searchQuery.length < 2) {
      $searchCount.textContent = '';
      $searchResults.classList.remove('active');
      $searchResults.innerHTML = '';
      searchResults = [];
      if (activeIndex >= 0) renderTextView(ocrData[activeIndex]);
      return;
    }
    performSearch(searchQuery);
  }, 300);
});

function performSearch(query) {
  api('GET', '/jobs/' + JOB_ID + '/search?q=' + encodeURIComponent(query)).then(function(results) {
    searchResults = results;
    $searchCount.textContent = results.length + ' result' + (results.length === 1 ? '' : 's');

    if (results.length === 0) {
      $searchResults.classList.remove('active');
      $searchResults.innerHTML = '';
      return;
    }

    var html = '';
    results.forEach(function(r, idx) {
      r.snippets.forEach(function(snip) {
        var before = escapeHtml(snip.text.substring(0, snip.match_start));
        var match = escapeHtml(snip.text.substring(snip.match_start, snip.match_end));
        var after = escapeHtml(snip.text.substring(snip.match_end));
        html += '<div class="search-result-item" data-result-idx="' + idx + '">'
          + '<span class="search-result-time">' + formatTime(r.timestamp_ms) + '</span>'
          + before + '<b>' + match + '</b>' + after
          + '</div>';
      });
    });

    $searchResults.innerHTML = html;
    $searchResults.classList.add('active');

    $searchResults.querySelectorAll('.search-result-item').forEach(function(item) {
      item.addEventListener('click', function() {
        var r = searchResults[parseInt(item.dataset.resultIdx)];
        for (var i = 0; i < ocrData.length; i++) {
          if (ocrData[i].screenshot_id === r.id || ocrData[i].frame_ref === r.frame_ref) {
            setActiveItem(i);
            break;
          }
        }
      });
    });

    if (activeIndex >= 0) renderTextView(ocrData[activeIndex]);
  });
}

// ---- Run OCR ----
function runOcr(force) {
  $processingOverlay.classList.add('active');
  $processingProgress.textContent = force
    ? 'Re-processing all frames with enhanced OCR pipeline...'
    : 'Processing frames with enhanced OCR pipeline...';

  var url = '/jobs/' + JOB_ID + '/run-ocr' + (force ? '?force=true' : '');
  api('POST', url).then(function(result) {
    var tableCount = 0;
    if (result.results) {
      result.results.forEach(function(r) { tableCount += r.table_count || 0; });
    }
    $processingProgress.textContent = 'Done! Processed ' + result.processed + ' frames'
      + (tableCount > 0 ? ', found ' + tableCount + ' table(s)' : '') + '.';
    setTimeout(function() {
      $processingOverlay.classList.remove('active');
      loadData();
    }, 1200);
  }).catch(function(err) {
    $processingProgress.textContent = 'Error: ' + err.message;
    setTimeout(function() { $processingOverlay.classList.remove('active'); }, 3000);
  });
}

document.getElementById('btn-run-ocr').addEventListener('click', function() {
  runOcr(false);
});

// Re-run OCR button (if present)
var $btnRerunOcr = document.getElementById('btn-rerun-ocr');
if ($btnRerunOcr) {
  $btnRerunOcr.addEventListener('click', function() {
    runOcr(true);
  });
}

// Double-click Run OCR = force re-run
document.getElementById('btn-run-ocr').addEventListener('dblclick', function(e) {
  e.preventDefault();
  runOcr(true);
});

// ---- Navigation ----
document.getElementById('btn-back-frames').addEventListener('click', function() {
  window.location.href = '/jobs/' + JOB_ID + '/frames';
});

document.getElementById('btn-export').addEventListener('click', function() {
  var exportData = ocrData.map(function(item) {
    return {
      timestamp: formatTime(item.timestamp_ms),
      timestamp_ms: item.timestamp_ms,
      extracted_text: item.extracted_text,
      entities: item.entities,
      tables: (item.entities && item.entities.tables) ? item.entities.tables : [],
      notes: item.notes,
      section_type: item.section_type,
      ocr_confidence: item.ocr_confidence,
    };
  });

  var blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'ocr_export_' + JOB_ID.substring(0, 8) + '.json';
  a.click();
  URL.revokeObjectURL(url);
});

// ---- Keyboard nav ----
document.addEventListener('keydown', function(e) {
  if (e.target === $searchInput || e.target === $notesTextarea) return;

  if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
    e.preventDefault();
    if (activeIndex < ocrData.length - 1) setActiveItem(activeIndex + 1);
  } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
    e.preventDefault();
    if (activeIndex > 0) setActiveItem(activeIndex - 1);
  } else if (e.key === 'f' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    $searchInput.focus();
  }
});

// ---- Init ----
loadData();
