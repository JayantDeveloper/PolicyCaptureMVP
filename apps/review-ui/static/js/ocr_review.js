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
var confidenceThreshold = 0;
var selectedFrames = {};  // index -> true
var selectionMode = false;

// Entity type config
var ENTITY_TYPES = {
  date:           { label: 'Dates',          color: '#f59e0b' },
  time_value:     { label: 'Times',          color: '#fbbf24' },
  currency:       { label: 'Currency',       color: '#22c55e' },
  email:          { label: 'Emails',         color: '#3b82f6' },
  phone:          { label: 'Phone',          color: '#a78bfa' },
  percentage:     { label: 'Percentages',    color: '#14b8a6' },
  person_name:    { label: 'People',         color: '#f472b6' },
  organization:   { label: 'Organizations',  color: '#fb923c' },
  address:        { label: 'Addresses',      color: '#818cf8' },
  state:          { label: 'States',         color: '#a3e635' },
  zip_code:       { label: 'ZIP Codes',      color: '#86efac' },
  policy_number:  { label: 'Policy #',       color: '#e879f9' },
  case_number:    { label: 'Case #',         color: '#67e8f9' },
  claim_number:   { label: 'Claims',         color: '#7dd3fc' },
  group_number:   { label: 'Group #',        color: '#c084fc' },
  ssn:            { label: 'SSN',            color: '#ef4444' },
  ein:            { label: 'EIN/Tax ID',     color: '#f87171' },
  npi:            { label: 'NPI',            color: '#fb7185' },
  medical_code:   { label: 'Medical Codes',  color: '#34d399' },
  account_number: { label: 'Accounts',       color: '#fda4af' },
  url:            { label: 'URLs',           color: '#94a3b8' },
  id_number:      { label: 'IDs',            color: '#d4d4d8' },
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
var $globalNerView = document.getElementById('global-ner-view');
var $selCount = document.getElementById('sel-count');
var $selClear = document.getElementById('sel-clear');

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
    showThresholdControl();
    if (confidenceThreshold > 0) applyThresholdFilter();
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
  var totalFormFields = 0;
  var totalLists = 0;
  ocrData.forEach(function(item) {
    if (item.entities && item.entities.summary) {
      totalEntities += item.entities.summary.total_entities || 0;
      totalFormFields += item.entities.summary.form_fields_found || 0;
      totalLists += item.entities.summary.lists_found || 0;
    }
    if (item.entities && item.entities.tables) {
      totalTables += item.entities.tables.length;
    }
  });
  document.getElementById('stat-entities').innerHTML = 'Entities: <b>' + totalEntities + '</b>';
  var dataItems = totalTables + totalFormFields + totalLists;
  document.getElementById('stat-tables').innerHTML = 'Data: <b>' + dataItems + '</b>'
    + (totalTables > 0 ? ' (' + totalTables + ' table' + (totalTables > 1 ? 's' : '') + ')' : '');
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

    var formCount = item.entities && item.entities.summary ? item.entities.summary.form_fields_found || 0 : 0;
    var listCount = item.entities && item.entities.summary ? item.entities.summary.lists_found || 0 : 0;

    var metaParts = [];
    if (hasText) metaParts.push(item.extracted_text.length + ' chars');
    else metaParts.push('No OCR');
    if (entCount > 0) metaParts.push(entCount + ' entities');
    if (tableCount > 0) metaParts.push(tableCount + ' table' + (tableCount > 1 ? 's' : ''));
    if (formCount > 0) metaParts.push(formCount + ' field' + (formCount > 1 ? 's' : ''));
    if (listCount > 0) metaParts.push(listCount + ' list' + (listCount > 1 ? 's' : ''));

    var selClass = selectedFrames[idx] ? ' selected' : '';
    html += '<div class="fs-card' + (idx === activeIndex ? ' active' : '') + (hasText ? ' has-text' : ' no-text') + selClass + '" data-index="' + idx + '">'
      + '<div class="fs-check">\u2713</div>'
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
    card.querySelector('.fs-check').addEventListener('click', function(e) {
      e.stopPropagation();
      var idx = parseInt(card.dataset.index);
      toggleFrameSelection(idx);
    });
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

// ---- Tables & Structured Data view ----
function renderTablesView(item) {
  var tables = (item.entities && item.entities.tables) ? item.entities.tables : [];
  var checkboxes = (item.entities && item.entities.checkboxes) ? item.entities.checkboxes : [];
  var formFields = (item.entities && item.entities.form_fields) ? item.entities.form_fields : [];
  var formData = (item.entities && item.entities.form_data) ? item.entities.form_data : {};
  var kvPairs = formData.key_value_pairs || [];
  var textLists = formData.lists || [];
  var sectionHeaders = formData.section_headers || [];

  var hasContent = tables.length > 0 || checkboxes.length > 0 || formFields.length > 0
    || kvPairs.length > 0 || textLists.length > 0 || sectionHeaders.length > 0;

  if (!hasContent) {
    $tablesView.innerHTML = '<div class="empty-state"><p>No structured data detected</p>'
      + '<p style="font-size:11px;color:#475569">Tables, forms, lists, and key-value pairs are automatically detected</p></div>';
    return;
  }

  var html = '';

  // --- Section Headers ---
  if (sectionHeaders.length > 0) {
    html += '<div class="table-card">'
      + '<div class="table-card-header">'
      + '<span class="table-card-title" style="color:#a78bfa">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h10M4 18h14"/></svg>'
      + 'Section Headers'
      + '</span>'
      + '<span class="table-card-meta">' + sectionHeaders.length + ' found</span>'
      + '</div>'
      + '<div style="padding:8px 12px">';
    sectionHeaders.forEach(function(h) {
      html += '<div style="padding:4px 8px;margin:2px 0;background:#1e1b4b;border-radius:4px;font-weight:600;color:#c4b5fd;font-size:13px">'
        + escapeHtml(h) + '</div>';
    });
    html += '</div></div>';
  }

  // --- Key-Value Pairs (Form Data) ---
  if (kvPairs.length > 0) {
    html += '<div class="table-card">'
      + '<div class="table-card-header">'
      + '<span class="table-card-title" style="color:#22d3ee">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/><path d="M9 12h6M9 16h6"/></svg>'
      + 'Form Fields'
      + '</span>'
      + '<span class="table-card-meta">' + kvPairs.length + ' fields</span>'
      + '</div>';

    html += '<table class="extracted-table">'
      + '<thead><tr><th style="width:35%">Field</th><th>Value</th></tr></thead><tbody>';
    kvPairs.forEach(function(kv) {
      var knownBadge = kv.is_known_field
        ? ' <span style="font-size:9px;background:#065f46;color:#6ee7b7;padding:1px 5px;border-radius:3px;margin-left:4px">known</span>'
        : '';
      html += '<tr><td style="font-weight:600;color:#94a3b8">' + escapeHtml(kv.key) + knownBadge + '</td>'
        + '<td>' + escapeHtml(kv.value) + '</td></tr>';
    });
    html += '</tbody></table></div>';
  }

  // --- Tables ---
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

  // --- Lists ---
  if (textLists.length > 0) {
    textLists.forEach(function(lst, li) {
      var icon = lst.type === 'numbered'
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 6h11M10 12h11M10 18h11M4 6h1v4M4 10h2M6 18H4c0-1 2-2 2-3s-1-1.5-2-1"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 6h9M11 12h9M11 18h9"/><circle cx="5" cy="6" r="1.5"/><circle cx="5" cy="12" r="1.5"/><circle cx="5" cy="18" r="1.5"/></svg>';

      html += '<div class="table-card">'
        + '<div class="table-card-header">'
        + '<span class="table-card-title" style="color:#fb923c">'
        + icon
        + (lst.type === 'numbered' ? 'Numbered' : lst.type === 'lettered' ? 'Lettered' : 'Bulleted') + ' List'
        + '</span>'
        + '<span class="table-card-meta">' + lst.items.length + ' items</span>'
        + '</div>'
        + '<div style="padding:8px 12px">';

      var tag = lst.type === 'numbered' ? 'ol' : 'ul';
      html += '<' + tag + ' style="margin:0;padding-left:24px;color:#e2e8f0;font-size:13px;line-height:1.8">';
      lst.items.forEach(function(item) {
        html += '<li>' + escapeHtml(item) + '</li>';
      });
      html += '</' + tag + '></div></div>';
    });
  }

  // --- Checkboxes ---
  if (checkboxes.length > 0) {
    html += '<div class="table-card">'
      + '<div class="table-card-header">'
      + '<span class="table-card-title" style="color:#4ade80">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 12l2 2 4-4"/></svg>'
      + 'Checkboxes / Selections'
      + '</span>'
      + '<span class="table-card-meta">' + checkboxes.length + ' detected</span>'
      + '</div>'
      + '<div style="padding:8px 12px;display:flex;flex-wrap:wrap;gap:6px">';

    checkboxes.forEach(function(cb) {
      var icon = cb.checked ? '\u2611' : '\u2610';
      var color = cb.checked ? '#4ade80' : '#94a3b8';
      html += '<span style="padding:3px 8px;background:#1e293b;border-radius:4px;font-size:13px;color:' + color + '">'
        + icon + ' ' + cb.type
        + '</span>';
    });
    html += '</div></div>';
  }

  // --- Form Input Fields (visual) ---
  if (formFields.length > 0) {
    html += '<div class="table-card">'
      + '<div class="table-card-header">'
      + '<span class="table-card-title" style="color:#60a5fa">'
      + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10v4"/></svg>'
      + 'Input Fields (Visual)'
      + '</span>'
      + '<span class="table-card-meta">' + formFields.length + ' detected</span>'
      + '</div>'
      + '<div style="padding:8px 12px;display:flex;flex-wrap:wrap;gap:6px">';

    formFields.forEach(function(ff) {
      html += '<span style="padding:3px 8px;background:#1e293b;border-radius:4px;font-size:12px;color:#60a5fa">'
        + ff.field_type.replace(/_/g, ' ') + ' (' + ff.bbox[2] + '&times;' + ff.bbox[3] + 'px)'
        + '</span>';
    });
    html += '</div></div>';
  }

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
    $globalNerView.style.display = currentTab === 'global-ner' ? 'flex' : 'none';
    $notesView.style.display = currentTab === 'notes' ? 'flex' : 'none';

    if (currentTab === 'global-ner') renderGlobalNer();

    // Update title
    var titles = { text: 'Extracted Text', entities: 'Named Entities', tables: 'Structured Data', 'global-ner': 'Global NER', notes: 'Annotations' };
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

// ---- Export dropdown ----
var $exportMenu = document.getElementById('export-menu');
document.getElementById('btn-export').addEventListener('click', function(e) {
  e.stopPropagation();
  $exportMenu.classList.toggle('open');
});
document.addEventListener('click', function() { $exportMenu.classList.remove('open'); });

function getSelectedData() {
  var hasSelection = Object.keys(selectedFrames).length > 0;
  var source = hasSelection
    ? ocrData.filter(function(_, idx) { return selectedFrames[idx]; })
    : ocrData;
  return source;
}

function getExportData() {
  return getSelectedData().map(function(item) {
    var ent = item.entities || {};
    return {
      timestamp: formatTime(item.timestamp_ms),
      timestamp_ms: item.timestamp_ms,
      extracted_text: item.extracted_text,
      entities: ent.entities || [],
      categories: ent.categories || {},
      tables: ent.tables || [],
      form_data: ent.form_data || {},
      checkboxes: ent.checkboxes || [],
      form_fields: ent.form_fields || [],
      notes: item.notes,
      section_type: item.section_type,
      ocr_confidence: item.ocr_confidence,
    };
  });
}

function exportJson() {
  var blob = new Blob([JSON.stringify(getExportData(), null, 2)], { type: 'application/json' });
  downloadBlob(blob, 'ocr_export_' + JOB_ID.substring(0, 8) + '.json');
}

function exportTxt() {
  var lines = [];
  getSelectedData().forEach(function(item) {
    var time = formatTime(item.timestamp_ms);
    var conf = item.ocr_confidence ? ' (confidence: ' + Math.round(item.ocr_confidence) + '%)' : '';
    lines.push('=== Frame at ' + time + conf + ' ===');
    lines.push('');
    if (item.extracted_text) {
      lines.push(item.extracted_text);
    } else {
      lines.push('[No text extracted]');
    }
    lines.push('');

    var ent = item.entities || {};
    // Entities summary
    if (ent.categories) {
      var entParts = [];
      Object.keys(ent.categories).forEach(function(type) {
        var vals = ent.categories[type];
        if (vals && vals.length > 0) {
          entParts.push(type.replace(/_/g, ' ') + ': ' + vals.join(', '));
        }
      });
      if (entParts.length > 0) {
        lines.push('--- Entities ---');
        entParts.forEach(function(p) { lines.push('  ' + p); });
        lines.push('');
      }
    }

    // Key-value pairs
    var formData = ent.form_data || {};
    var kvPairs = formData.key_value_pairs || [];
    if (kvPairs.length > 0) {
      lines.push('--- Form Fields ---');
      kvPairs.forEach(function(kv) { lines.push('  ' + kv.key + ': ' + kv.value); });
      lines.push('');
    }

    // Tables
    var tables = ent.tables || [];
    tables.forEach(function(table, ti) {
      lines.push('--- Table ' + (ti + 1) + ' ---');
      if (table.rows) {
        table.rows.forEach(function(row) { lines.push('  ' + row.join(' | ')); });
      } else if (table.raw_text) {
        lines.push(table.raw_text);
      }
      lines.push('');
    });

    // Notes
    if (item.notes) {
      lines.push('--- Notes ---');
      lines.push(item.notes);
      lines.push('');
    }

    lines.push('');
  });
  var blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  downloadBlob(blob, 'ocr_export_' + JOB_ID.substring(0, 8) + '.txt');
}

function exportDocx() {
  // If frames are selected, export client-side subset via backend with indices
  var hasSelection = Object.keys(selectedFrames).length > 0;
  var url = API + '/jobs/' + JOB_ID + '/export-docx';
  if (hasSelection) {
    url += '?indices=' + Object.keys(selectedFrames).join(',');
  }
  fetch(url, { method: 'POST' }).then(function(res) {
    if (!res.ok) return res.json().then(function(e) { throw new Error(e.detail || res.statusText); });
    return res.blob();
  }).then(function(blob) {
    downloadBlob(blob, 'ocr_export_' + JOB_ID.substring(0, 8) + '.docx');
  }).catch(function(err) {
    alert('DOCX export failed: ' + err.message);
  });
}

function downloadBlob(blob, filename) {
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

$exportMenu.querySelectorAll('.export-menu-item').forEach(function(item) {
  item.addEventListener('click', function(e) {
    e.stopPropagation();
    $exportMenu.classList.remove('open');
    var fmt = item.dataset.format;
    if (fmt === 'json') exportJson();
    else if (fmt === 'txt') exportTxt();
    else if (fmt === 'docx') exportDocx();
  });
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

// ---- Frame selection system ----
function toggleFrameSelection(idx) {
  if (selectedFrames[idx]) {
    delete selectedFrames[idx];
  } else {
    selectedFrames[idx] = true;
  }
  updateSelectionUI();
}

function setSelection(indices) {
  selectedFrames = {};
  indices.forEach(function(i) { selectedFrames[i] = true; });
  updateSelectionUI();
}

function clearSelection() {
  selectedFrames = {};
  updateSelectionUI();
}

function updateSelectionUI() {
  var count = Object.keys(selectedFrames).length;
  selectionMode = count > 0;

  // Toggle selection-mode class on frame list for showing checkboxes
  $frameList.parentElement.classList.toggle('selection-mode', selectionMode);

  // Update card classes
  $frameList.querySelectorAll('.fs-card').forEach(function(card) {
    var idx = parseInt(card.dataset.index);
    card.classList.toggle('selected', !!selectedFrames[idx]);
  });

  // Update count display
  if (count > 0) {
    $selCount.innerHTML = '<b>' + count + '</b> selected';
    $selClear.style.display = 'inline-flex';
  } else {
    $selCount.textContent = '';
    $selClear.style.display = 'none';
  }
}

// Select All visible frames
document.getElementById('sel-all').addEventListener('click', function() {
  var indices = [];
  $frameList.querySelectorAll('.fs-card').forEach(function(card) {
    if (!card.classList.contains('filtered-out')) {
      indices.push(parseInt(card.dataset.index));
    }
  });
  setSelection(indices);
});

// Select by Filter (confidence threshold - uses current slider value or prompts)
document.getElementById('sel-filter').addEventListener('click', function() {
  var minConf = confidenceThreshold > 0 ? confidenceThreshold : 50;
  var indices = [];
  ocrData.forEach(function(item, idx) {
    var conf = item.ocr_confidence || 0;
    if (conf >= minConf) indices.push(idx);
  });
  setSelection(indices);
  // Show threshold control if hidden
  $thresholdControl.style.display = 'block';
  $thresholdSlider.value = minConf;
  $thresholdValue.textContent = minConf + '%';
  confidenceThreshold = minConf;
  applyThresholdFilter();
});

// Metric dropdown toggle
var $metricMenu = document.getElementById('metric-menu');
document.getElementById('sel-metric').addEventListener('click', function(e) {
  e.stopPropagation();
  $metricMenu.classList.toggle('open');
});
document.addEventListener('click', function() { $metricMenu.classList.remove('open'); });

// Metric computations
function computeFrameMetrics(metric) {
  var scores = [];

  if (metric === 'density') {
    // Information density: text_length * entity_count * (confidence/100)
    ocrData.forEach(function(item, idx) {
      var textLen = (item.extracted_text || '').length;
      var entCount = (item.entities && item.entities.summary) ? item.entities.summary.total_entities || 0 : 0;
      var conf = (item.ocr_confidence || 0) / 100;
      var score = textLen * Math.max(entCount, 1) * Math.max(conf, 0.1);
      scores.push({ idx: idx, score: score });
    });
  } else if (metric === 'data-rich') {
    // Frames with tables, forms, KV pairs, lists
    ocrData.forEach(function(item, idx) {
      var ent = item.entities || {};
      var tableCount = (ent.tables || []).length;
      var formData = ent.form_data || {};
      var kvCount = (formData.key_value_pairs || []).length;
      var listCount = (formData.lists || []).length;
      var cbCount = (ent.checkboxes || []).length;
      var ffCount = (ent.form_fields || []).length;
      var score = tableCount * 10 + kvCount * 3 + listCount * 2 + cbCount + ffCount;
      scores.push({ idx: idx, score: score });
    });
  } else if (metric === 'entity-diverse') {
    // Most unique entity types per frame
    ocrData.forEach(function(item, idx) {
      var types = 0;
      if (item.entities && item.entities.categories) {
        Object.keys(item.entities.categories).forEach(function(t) {
          if (item.entities.categories[t] && item.entities.categories[t].length > 0) types++;
        });
      }
      scores.push({ idx: idx, score: types });
    });
  } else if (metric === 'unique-content') {
    // Jaccard distance: lower overlap with other frames = more unique
    var allWords = ocrData.map(function(item) {
      var text = (item.extracted_text || '').toLowerCase();
      var words = {};
      text.split(/\s+/).forEach(function(w) {
        if (w.length > 2) words[w] = true;
      });
      return words;
    });
    ocrData.forEach(function(item, idx) {
      var myWords = allWords[idx];
      var myKeys = Object.keys(myWords);
      if (myKeys.length === 0) { scores.push({ idx: idx, score: 0 }); return; }
      var totalOverlap = 0;
      var comparisons = 0;
      allWords.forEach(function(otherWords, oidx) {
        if (oidx === idx) return;
        var otherKeys = Object.keys(otherWords);
        if (otherKeys.length === 0) return;
        var intersection = 0;
        myKeys.forEach(function(w) { if (otherWords[w]) intersection++; });
        var union = myKeys.length + otherKeys.length - intersection;
        totalOverlap += union > 0 ? intersection / union : 0;
        comparisons++;
      });
      var avgOverlap = comparisons > 0 ? totalOverlap / comparisons : 0;
      // Invert: less overlap = higher score, weighted by having actual content
      scores.push({ idx: idx, score: (1 - avgOverlap) * myKeys.length });
    });
  }

  return scores;
}

$metricMenu.querySelectorAll('.metric-menu-item').forEach(function(item) {
  item.addEventListener('click', function(e) {
    e.stopPropagation();
    $metricMenu.classList.remove('open');
    var metric = item.dataset.metric;
    var scores = computeFrameMetrics(metric);

    // Select top 50% of frames with score > 0, sorted by score descending
    scores.sort(function(a, b) { return b.score - a.score; });
    var nonZero = scores.filter(function(s) { return s.score > 0; });
    var cutoff = Math.max(1, Math.ceil(nonZero.length * 0.5));
    var selected = nonZero.slice(0, cutoff).map(function(s) { return s.idx; });
    setSelection(selected);
  });
});

// Clear selection
$selClear.addEventListener('click', function() {
  clearSelection();
});

// ---- Global NER ----
function buildGlobalNerData() {
  // Aggregate all entities across all frames (or selected frames)
  var source = getSelectedData();
  var entityMap = {};  // "type|value" -> { type, text, frames: [idx, ...], count }
  var typeTotals = {};

  source.forEach(function(item, localIdx) {
    // Find the real index in ocrData
    var realIdx = ocrData.indexOf(item);
    if (!item.entities || !item.entities.categories) return;

    Object.keys(item.entities.categories).forEach(function(type) {
      var vals = item.entities.categories[type];
      if (!vals || vals.length === 0) return;

      vals.forEach(function(val) {
        var key = type + '|' + val;
        if (!entityMap[key]) {
          entityMap[key] = { type: type, text: val, frames: [], count: 0 };
        }
        if (entityMap[key].frames.indexOf(realIdx) === -1) {
          entityMap[key].frames.push(realIdx);
        }
        entityMap[key].count++;
        typeTotals[type] = (typeTotals[type] || 0) + 1;
      });
    });
  });

  // Convert to array and sort by frequency
  var entities = Object.keys(entityMap).map(function(k) { return entityMap[k]; });
  entities.sort(function(a, b) { return b.count - a.count; });

  return { entities: entities, typeTotals: typeTotals, frameCount: source.length };
}

var globalNerSearch = '';

function renderGlobalNer() {
  var data = buildGlobalNerData();
  var entities = data.entities;
  var typeTotals = data.typeTotals;

  if (entities.length === 0) {
    $globalNerView.innerHTML = '<div class="empty-state"><p>No entities found across frames</p>'
      + '<p style="font-size:11px;color:#475569">Run OCR first, then entities will be aggregated here</p></div>';
    return;
  }

  // Summary row
  var totalUnique = entities.length;
  var totalMentions = 0;
  entities.forEach(function(e) { totalMentions += e.count; });
  var typeCount = Object.keys(typeTotals).length;
  var hasSelection = Object.keys(selectedFrames).length > 0;

  var html = '';

  // Search
  html += '<div class="ner-search-wrap">'
    + '<svg class="ner-search-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>'
    + '<input class="ner-search" id="ner-search-input" type="text" placeholder="Search entities..." value="' + escapeHtml(globalNerSearch) + '">'
    + '</div>';

  // Summary chips
  html += '<div class="ner-summary-row">'
    + '<span class="ner-summary-chip"><b>' + totalUnique + '</b> unique</span>'
    + '<span class="ner-summary-chip"><b>' + totalMentions + '</b> mentions</span>'
    + '<span class="ner-summary-chip"><b>' + typeCount + '</b> types</span>'
    + '<span class="ner-summary-chip"><b>' + data.frameCount + '</b> frames' + (hasSelection ? ' (selected)' : '') + '</span>'
    + '</div>';

  // Group by entity type
  var grouped = {};
  entities.forEach(function(e) {
    if (!grouped[e.type]) grouped[e.type] = [];
    grouped[e.type].push(e);
  });

  Object.keys(ENTITY_TYPES).forEach(function(type) {
    var group = grouped[type];
    if (!group || group.length === 0) return;

    // Apply search filter
    var filtered = group;
    if (globalNerSearch) {
      var q = globalNerSearch.toLowerCase();
      filtered = group.filter(function(e) { return e.text.toLowerCase().indexOf(q) >= 0; });
    }
    if (filtered.length === 0) return;

    var cfg = ENTITY_TYPES[type];
    html += '<div class="entity-group">'
      + '<div class="entity-group-header ent-' + type + '">'
      + '<span>' + cfg.label + '</span>'
      + '<span class="entity-group-count">' + filtered.length + ' unique / ' + (typeTotals[type] || 0) + ' total</span>'
      + '</div>'
      + '<div style="padding:6px 4px">';

    filtered.forEach(function(ent, ei) {
      var frameId = 'ner-frames-' + type + '-' + ei;
      html += '<div class="ner-entity-row" data-frame-id="' + frameId + '">'
        + '<span class="ner-entity-text" style="color:' + cfg.color + '">' + escapeHtml(ent.text) + '</span>'
        + '<span class="ner-entity-freq">' + ent.count + 'x in ' + ent.frames.length + ' frame' + (ent.frames.length > 1 ? 's' : '') + '</span>'
        + '</div>';
      html += '<div class="ner-entity-frames" id="' + frameId + '">';
      ent.frames.forEach(function(fIdx) {
        var time = formatTime(ocrData[fIdx].timestamp_ms);
        html += '<span class="ner-frame-link" data-frame-idx="' + fIdx + '">' + time + '</span>';
      });
      html += '</div>';
    });

    html += '</div></div>';
  });

  $globalNerView.innerHTML = html;

  // Wire up search
  var $nerSearch = document.getElementById('ner-search-input');
  if ($nerSearch) {
    $nerSearch.addEventListener('input', function() {
      globalNerSearch = $nerSearch.value.trim();
      renderGlobalNer();
      // Re-focus and restore cursor position
      var newInput = document.getElementById('ner-search-input');
      if (newInput) {
        newInput.focus();
        newInput.setSelectionRange(newInput.value.length, newInput.value.length);
      }
    });
  }

  // Wire up expand/collapse for frame links
  $globalNerView.querySelectorAll('.ner-entity-row').forEach(function(row) {
    row.addEventListener('click', function() {
      var frameEl = document.getElementById(row.dataset.frameId);
      if (frameEl) frameEl.classList.toggle('open');
    });
  });

  // Wire up frame navigation links
  $globalNerView.querySelectorAll('.ner-frame-link').forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.stopPropagation();
      var idx = parseInt(link.dataset.frameIdx);
      setActiveItem(idx);
    });
  });
}

// ---- Confidence threshold ----
var $thresholdControl = document.getElementById('threshold-control');
var $thresholdSlider = document.getElementById('threshold-slider');
var $thresholdValue = document.getElementById('threshold-value');
var $thresholdShowing = document.getElementById('threshold-showing');

$thresholdSlider.addEventListener('input', function() {
  confidenceThreshold = parseInt($thresholdSlider.value);
  $thresholdValue.textContent = confidenceThreshold + '%';
  applyThresholdFilter();
});

function applyThresholdFilter() {
  var cards = $frameList.querySelectorAll('.fs-card');
  var shown = 0;
  var total = cards.length;
  cards.forEach(function(card) {
    var idx = parseInt(card.dataset.index);
    var item = ocrData[idx];
    var conf = item ? (item.ocr_confidence || 0) : 0;
    if (confidenceThreshold > 0 && conf < confidenceThreshold) {
      card.classList.add('filtered-out');
    } else {
      card.classList.remove('filtered-out');
      shown++;
    }
  });
  $thresholdShowing.textContent = shown + ' of ' + total + ' frames shown';
}

function showThresholdControl() {
  // Only show if OCR has been run (at least some items have confidence)
  var hasConf = ocrData.some(function(item) { return (item.ocr_confidence || 0) > 0; });
  $thresholdControl.style.display = hasConf ? 'block' : 'none';
}

// ---- Init ----
loadData();
