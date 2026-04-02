/* ============================================================
   PolicyCapture Local - Review UI Application
   Vanilla JS SPA with hash-based routing
   ============================================================ */

// --------------- API Client ---------------

const API = {
  base: '',

  async request(method, path, body = null) {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body && method !== 'GET') {
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(`${this.base}${path}`, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Request failed: ${res.status}`);
    }
    return res.json();
  },

  getJobs()                     { return this.request('GET', '/api/jobs'); },
  getJob(id)                    { return this.request('GET', `/api/jobs/${id}`); },
  createJob(title, videoPath)   { return this.request('POST', '/api/jobs', { title, source_video_path: videoPath || null }); },
  registerVideo(id, path)       { return this.request('POST', `/api/jobs/${id}/register-video`, { source_video_path: path }); },
  startProcessing(id)           { return this.request('POST', `/api/jobs/${id}/process`); },
  getScreenshots(id)            { return this.request('GET', `/api/jobs/${id}/screenshots`); },
  getSections(id)               { return this.request('GET', `/api/jobs/${id}/sections`); },
  updateScreenshot(id, data)    { return this.request('PATCH', `/api/screenshots/${id}`, data); },
  generateReport(id)            { return this.request('POST', `/api/jobs/${id}/report`); },
  getReport(id)                 { return this.request('GET', `/api/jobs/${id}/report`); },
  getReportHtml(id)             { return fetch(`/api/jobs/${id}/report/html`).then(r => r.ok ? r.text() : Promise.reject(new Error('No report'))); },
  deleteJob(id)                  { return this.request('DELETE', `/api/jobs/${id}`); },
  updateJobTitle(id, title)      { return this.request('PATCH', `/api/jobs/${id}/title`, { title }); },
  autoTitle(id)                  { return this.request('POST', `/api/jobs/${id}/auto-title`); },
  backfillConfidence(id)         { return this.request('POST', `/api/jobs/${id}/backfill-confidence`); },
  updateJobMetadata(id, data)    { return this.request('PATCH', `/api/jobs/${id}/metadata`, data); },
  seedDemo()                    { return this.request('POST', '/api/demo/seed'); },

  async uploadVideo(jobId, file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${this.base}/api/jobs/${jobId}/upload`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },
};

// --------------- Toast Notifications ---------------

function showToast(message, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// --------------- Rendering Helpers ---------------

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'className') node.className = v;
    else if (k === 'innerHTML') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v);
  }
  for (const c of Array.isArray(children) ? children : [children]) {
    if (typeof c === 'string') node.appendChild(document.createTextNode(c));
    else if (c) node.appendChild(c);
  }
  return node;
}

function setApp(content) {
  const app = document.getElementById('app');
  if (typeof content === 'string') {
    app.innerHTML = content;
  } else {
    app.innerHTML = '';
    app.appendChild(content);
  }
}

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function statusBadge(status) {
  const cls = {
    pending: 'badge-pending',
    processing: 'badge-processing',
    completed: 'badge-completed',
    failed: 'badge-failed',
  }[status] || 'badge-pending';
  return el('span', { className: `badge ${cls}` }, [status || 'pending']);
}

// --------------- Page: Job List ---------------

async function loadJobs() {
  setApp('<div class="p-6 text-muted text-sm">Loading jobs...</div>');
  try {
    const jobs = await API.getJobs();
    renderJobList(jobs);
  } catch (err) {
    showToast(err.message, 'error');
    setApp('<div class="p-6 text-muted">Failed to load jobs.</div>');
  }
}

function renderJobList(jobs) {
  const header = el('div', { className: 'page-header' }, [
    el('h1', {}, ['Jobs']),
    el('div', { className: 'flex gap-2' }, [
      el('a', { className: 'btn btn-primary btn-sm', href: '/recorder', style: 'background:#dc2626;border-color:#dc2626;' }, ['Record Screen']),
      el('button', { className: 'btn btn-secondary btn-sm', onClick: openCreateModal }, ['+ New Job']),
      el('button', { className: 'btn btn-secondary btn-sm', onClick: handleSeedDemo }, ['Seed Demo']),
    ]),
  ]);

  const wrapper = el('div', { className: 'container p-6' });
  wrapper.appendChild(header);

  if (!jobs || jobs.length === 0) {
    wrapper.appendChild(
      el('div', { className: 'empty-state' }, [
        el('div', { className: 'empty-state-icon' }, ['--']),
        el('p', {}, ['No jobs yet. Create one to get started.']),
      ])
    );
  } else {
    const list = el('div', { className: 'job-list' });
    for (const job of jobs) {
      const titleWrap = el('div', { className: 'job-row-title-wrap' });
      const titleSpan = el('span', { className: 'job-row-title' }, [job.title || `Job ${job.id}`]);
      const renameBtn = el('button', {
        className: 'btn-rename',
        onClick: (e) => {
          e.preventDefault();
          e.stopPropagation();
          startInlineRename(titleWrap, job.id, job.title || `Job ${job.id}`);
        },
      }, ['Rename']);
      titleWrap.appendChild(titleSpan);
      titleWrap.appendChild(renameBtn);

      const row = el('a', {
        className: 'job-row',
        href: `#/jobs/${job.id}`,
      }, [
        el('div', { className: 'job-row-info' }, [
          titleWrap,
          el('div', { className: 'job-row-date' }, [formatDate(job.created_at)]),
        ]),
        statusBadge(job.status),
      ]);
      list.appendChild(row);
    }
    wrapper.appendChild(list);
  }

  setApp(wrapper);
}

// --------------- Modal: Create Job ---------------

function openCreateModal() {
  const backdrop = el('div', { className: 'modal-backdrop', onClick: (e) => {
    if (e.target === backdrop) backdrop.remove();
  }});

  let titleInput, pathInput;

  const modal = el('div', { className: 'modal' }, [
    el('h3', { className: 'mb-4' }, ['Create New Job']),
    el('div', { className: 'form-group mb-4' }, [
      el('label', { className: 'form-label' }, ['Title']),
      titleInput = el('input', { className: 'form-input', type: 'text', placeholder: 'e.g. 2024 Benefits Policy Review' }),
    ]),
    el('div', { className: 'form-group mb-4' }, [
      el('label', { className: 'form-label' }, ['Video Path (optional)']),
      pathInput = el('input', { className: 'form-input', type: 'text', placeholder: '/path/to/video.mp4' }),
    ]),
    el('div', { className: 'flex gap-2 justify-between' }, [
      el('button', { className: 'btn btn-secondary', onClick: () => backdrop.remove() }, ['Cancel']),
      el('button', { className: 'btn btn-primary', onClick: async () => {
        const title = titleInput.value.trim();
        if (!title) { showToast('Title is required', 'error'); return; }
        try {
          const job = await API.createJob(title, pathInput.value.trim() || null);
          backdrop.remove();
          showToast('Job created', 'success');
          window.location.hash = `#/jobs/${job.id}`;
        } catch (err) {
          showToast(err.message, 'error');
        }
      }}, ['Create']),
    ]),
  ]);

  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
  titleInput.focus();
}

// --------------- Inline Rename ---------------

function startInlineRename(wrapperEl, jobId, currentTitle) {
  // Replace the entire wrapper content with an input
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'inline-rename-input';
  input.value = currentTitle;

  // Save original HTML to restore on cancel
  const originalHTML = wrapperEl.innerHTML;
  wrapperEl.innerHTML = '';
  wrapperEl.appendChild(input);

  input.focus();
  input.select();

  let saved = false;
  async function save() {
    if (saved) return;
    saved = true;
    const newTitle = input.value.trim();
    if (newTitle && newTitle !== currentTitle) {
      try {
        await API.updateJobTitle(jobId, newTitle);
        showToast('Title updated', 'success');
      } catch (err) {
        showToast(err.message, 'error');
      }
    }
    route();
  }

  input.addEventListener('blur', save);
  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
    if (e.key === 'Escape') { saved = true; wrapperEl.innerHTML = originalHTML; }
  });
}

// --------------- Page: Job Detail ---------------

async function loadJobDetail(jobId) {
  setApp('<div class="p-6 text-muted text-sm">Loading job...</div>');
  try {
    const job = await API.getJob(jobId);
    renderJobDetail(job);
  } catch (err) {
    showToast(err.message, 'error');
    setApp('<div class="p-6 text-muted">Failed to load job.</div>');
  }
}

function renderJobDetail(job) {
  const wrapper = el('div', { className: 'container p-6' });

  // Back link
  wrapper.appendChild(
    el('a', { href: '#/', className: 'text-sm text-muted', style: 'text-decoration:none;' }, ['< Back to Jobs'])
  );

  // Header
  const titleWrap = el('div', { className: 'job-row-title-wrap' });
  const titleH1 = el('h1', {}, [job.title || `Job ${job.id}`]);
  const renameBtn = el('button', {
    className: 'btn-rename',
    onClick: () => startInlineRename(titleWrap, job.id, job.title || `Job ${job.id}`),
  }, ['Rename']);
  titleWrap.appendChild(titleH1);
  titleWrap.appendChild(renameBtn);

  const header = el('div', { className: 'page-header mt-4' }, [
    el('div', { className: 'flex items-center gap-3' }, [
      titleWrap,
      statusBadge(job.status),
    ]),
    el('div', { className: 'flex gap-2' }, [
      (job.status === 'completed' || job.frame_count > 0)
        ? el('a', { className: 'btn btn-primary btn-sm', href: `/jobs/${job.id}/frames` }, ['Review Frames'])
        : null,
      el('a', { className: 'btn btn-secondary btn-sm', href: `#/jobs/${job.id}/report` }, ['View Report']),
    ].filter(Boolean)),
  ]);
  wrapper.appendChild(header);

  // Info card
  const infoCard = el('div', { className: 'card mb-6' }, [
    el('div', { className: 'card-body' }, [
      el('div', { className: 'flex gap-6 flex-wrap' }, [
        el('div', { className: 'form-group' }, [
          el('span', { className: 'text-xs text-muted font-medium' }, ['Job ID']),
          el('span', { className: 'text-sm text-mono' }, [String(job.id)]),
        ]),
        el('div', { className: 'form-group' }, [
          el('span', { className: 'text-xs text-muted font-medium' }, ['Created']),
          el('span', { className: 'text-sm' }, [formatDate(job.created_at)]),
        ]),
        el('div', { className: 'form-group' }, [
          el('span', { className: 'text-xs text-muted font-medium' }, ['Video']),
          el('span', { className: 'text-sm text-mono' }, [job.source_video_path || 'None']),
        ]),
      ]),
    ]),
  ]);
  wrapper.appendChild(infoCard);

  // BAH Metadata card
  const metadataCard = buildMetadataCard(job);
  wrapper.appendChild(metadataCard);

  // Actions bar
  const actionsCard = el('div', { className: 'card mb-6' }, [
    el('div', { className: 'card-body flex gap-2 flex-wrap items-center' }, buildActionButtons(job)),
  ]);
  wrapper.appendChild(actionsCard);

  // Processing progress
  if (job.status === 'processing') {
    const progressSection = el('div', { className: 'card mb-6' }, [
      el('div', { className: 'card-body' }, [
        el('div', { className: 'flex items-center justify-between mb-2' }, [
          el('span', { className: 'text-sm font-medium' }, ['Processing...']),
        ]),
        el('div', { className: 'progress-bar-container' }, [
          el('div', { className: 'progress-bar indeterminate' }),
        ]),
      ]),
    ]);
    wrapper.appendChild(progressSection);
    startPolling(job.id);
  }

  // Screenshots section
  const screenshotsSection = el('div', { id: 'screenshots-section' });
  wrapper.appendChild(screenshotsSection);

  if (job.status === 'completed' || job.screenshot_count > 0) {
    loadScreenshots(job.id);
  }

  setApp(wrapper);
}

// --------------- Metadata Card ---------------

function buildMetadataCard(job) {
  const fields = [
    { key: 'recipient', label: 'Recipient', placeholder: 'e.g. Billy Summers', type: 'text' },
    { key: 'perm_id', label: 'PERM ID', placeholder: 'e.g. DCM2401F011', type: 'text' },
    { key: 'date_of_service', label: 'Date of Service', placeholder: 'MM/DD/YYYY', type: 'text' },
    { key: 'state', label: 'State', placeholder: 'e.g. DC', type: 'text' },
    { key: 'case_type', label: 'Case Type', placeholder: 'e.g. Medicaid', type: 'text' },
    { key: 'sample', label: 'Sample', placeholder: 'e.g. Sample A', type: 'text' },
  ];

  const inputs = {};
  const fieldEls = fields.map(f => {
    const input = el('input', {
      className: 'form-input form-input-sm',
      type: f.type,
      placeholder: f.placeholder,
    });
    input.value = job[f.key] || '';
    inputs[f.key] = input;
    return el('div', { className: 'metadata-field' }, [
      el('label', { className: 'form-label text-xs' }, [f.label]),
      input,
    ]);
  });

  let saveTimer;
  function autoSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      const data = {};
      for (const f of fields) {
        const val = inputs[f.key].value;
        if (val !== (job[f.key] || '')) {
          data[f.key] = val;
        }
      }
      if (Object.keys(data).length === 0) return;
      try {
        await API.updateJobMetadata(job.id, data);
        showToast('Metadata saved', 'success');
      } catch (err) {
        showToast(err.message, 'error');
      }
    }, 800);
  }

  for (const input of Object.values(inputs)) {
    input.addEventListener('input', autoSave);
  }

  const card = el('div', { className: 'card mb-6' }, [
    el('div', { className: 'card-body' }, [
      el('div', { className: 'flex items-center justify-between mb-3' }, [
        el('h3', { className: 'text-sm font-medium', style: 'margin:0;' }, ['Case Metadata']),
      ]),
      el('div', { className: 'metadata-grid' }, fieldEls),
    ]),
  ]);

  return card;
}

function buildActionButtons(job) {
  const buttons = [];

  if (!job.source_video_path) {
    const fileInput = el('input', { type: 'file', accept: 'video/*', style: 'display:none', onChange: async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        showToast('Uploading video...', 'info');
        await API.uploadVideo(job.id, file);
        showToast('Video uploaded', 'success');
        loadJobDetail(job.id);
      } catch (err) {
        showToast(err.message, 'error');
      }
    }});
    buttons.push(el('button', { className: 'btn btn-secondary btn-sm', onClick: () => fileInput.click() }, ['Upload Video']));
    buttons.push(fileInput);

    buttons.push(el('button', { className: 'btn btn-secondary btn-sm', onClick: () => {
      const path = prompt('Enter local video file path:');
      if (path) handleRegisterVideo(job.id, path);
    }}, ['Register Local Video']));
  }

  if (job.source_video_path && (job.status === 'pending' || job.status === 'failed')) {
    buttons.push(el('button', { className: 'btn btn-primary btn-sm', onClick: () => handleExtractFrames(job.id) }, ['Extract Frames']));
    buttons.push(el('button', { className: 'btn btn-secondary btn-sm', onClick: () => handleStartProcessing(job.id) }, ['Full Pipeline']));
  }

  if (job.status === 'completed') {
    buttons.push(el('button', { className: 'btn btn-success btn-sm', onClick: () => handleGenerateReport(job.id) }, ['Generate Report']));
    buttons.push(el('button', { className: 'btn btn-secondary btn-sm', onClick: () => handleAutoTitle(job.id) }, ['Auto-Title']));
  }

  // Always show delete
  buttons.push(el('button', {
    className: 'btn btn-sm',
    style: 'color:#ef4444;border-color:#ef4444;background:transparent;margin-left:auto;',
    onClick: () => handleDeleteJob(job.id),
  }, ['Delete']));

  return buttons;
}

// --------------- Screenshots ---------------

async function loadScreenshots(jobId) {
  const section = document.getElementById('screenshots-section');
  if (!section) return;
  section.innerHTML = '<div class="p-4 text-muted text-sm">Loading screenshots...</div>';

  try {
    const screenshots = await API.getScreenshots(jobId);
    renderScreenshotGallery(section, screenshots, jobId);
  } catch (err) {
    section.innerHTML = '<div class="p-4 text-muted">No screenshots available.</div>';
  }
}

function renderScreenshotGallery(container, screenshots, jobId) {
  container.innerHTML = '';

  const header = el('div', { className: 'flex items-center justify-between mb-4' }, [
    el('h2', {}, [`Screenshots (${screenshots.length})`]),
    el('div', { className: 'flex gap-2' }, [
      el('button', { className: 'btn btn-secondary btn-sm', onClick: () => loadScreenshots(jobId) }, ['Refresh']),
    ]),
  ]);
  container.appendChild(header);

  if (screenshots.length === 0) {
    container.appendChild(el('div', { className: 'empty-state' }, [
      el('p', {}, ['No screenshots extracted yet.']),
    ]));
    return;
  }

  const grid = el('div', { className: 'screenshot-grid' });
  for (const ss of screenshots) {
    grid.appendChild(renderScreenshotCard(ss));
  }
  container.appendChild(grid);
}

function renderScreenshotCard(ss) {
  const acceptedClass = ss.accepted === true ? ' accepted' : ss.accepted === false ? ' rejected' : '';

  const notesField = el('textarea', {
    className: 'screenshot-notes mt-2',
    placeholder: 'Add notes...',
  });
  notesField.value = ss.notes || '';

  let debounceTimer;
  notesField.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      handleUpdateScreenshot(ss.id, { notes: notesField.value });
    }, 600);
  });

  // Build image URL from the thumbnail or image path
  let imgSrc = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="280" height="180" fill="%23e2e8f0"><rect width="280" height="180"/></svg>';
  if (ss.thumbnail_path || ss.image_path) {
    const filePath = ss.thumbnail_path || ss.image_path;
    const parts = filePath.split('/');
    const jobIdx = parts.indexOf('jobs');
    if (jobIdx >= 0 && parts.length > jobIdx + 3) {
      const jobId = parts[jobIdx + 1];
      const artifactType = parts[jobIdx + 2];
      const filename = parts[parts.length - 1];
      imgSrc = `/api/artifacts/${jobId}/${artifactType}/${filename}`;
    }
  }

  const card = el('div', { className: `screenshot-card${acceptedClass}`, 'data-id': String(ss.id) }, [
    el('img', { src: imgSrc, alt: `Screenshot ${ss.order_index || ''}`, loading: 'lazy' }),
    el('div', { className: 'card-body' }, [
      el('div', { className: 'screenshot-meta' }, [
        el('div', { className: 'flex gap-2 items-center' }, [
          el('span', { className: 'text-xs text-muted' }, [`#${ss.order_index ?? ''}`]),
          ss.section_type
            ? el('span', { className: 'badge badge-section' }, [ss.section_type])
            : null,
        ].filter(Boolean)),
        el('div', { className: 'screenshot-actions' }, [
          el('button', {
            className: `btn btn-sm ${ss.accepted === true ? 'btn-success' : 'btn-secondary'}`,
            title: 'Accept',
            onClick: () => handleUpdateScreenshot(ss.id, { accepted: true }),
          }, ['Accept']),
          el('button', {
            className: `btn btn-sm ${ss.accepted === false ? 'btn-danger' : 'btn-secondary'}`,
            title: 'Reject',
            onClick: () => handleUpdateScreenshot(ss.id, { accepted: false }),
          }, ['Reject']),
        ]),
      ]),
      notesField,
    ]),
  ]);

  return card;
}

// --------------- Page: Report ---------------

async function loadReport(jobId) {
  setApp('<div class="container p-6"><div class="text-muted text-sm">Loading report...</div></div>');
  try {
    const html = await API.getReportHtml(jobId);
    renderReportPage(jobId, { html });
  } catch (err) {
    const wrapper = el('div', { className: 'container p-6' }, [
      el('a', { href: `#/jobs/${jobId}`, className: 'text-sm text-muted', style: 'text-decoration:none;' }, ['< Back to Job']),
      el('div', { className: 'empty-state mt-6' }, [
        el('p', {}, ['No report generated yet.']),
        el('button', { className: 'btn btn-primary btn-sm mt-4', onClick: () => handleGenerateReport(jobId) }, ['Generate Report']),
      ]),
    ]);
    setApp(wrapper);
  }
}

function renderReportPage(jobId, report) {
  const html = report.html || report.content || '<p>Report content unavailable.</p>';
  const wrapper = el('div', { className: 'container p-6' }, [
    el('a', { href: `#/jobs/${jobId}`, className: 'text-sm text-muted', style: 'text-decoration:none;' }, ['< Back to Job']),
    el('div', { className: 'page-header mt-4' }, [
      el('h1', {}, ['Evidence Report']),
      el('div', { className: 'flex gap-2' }, [
        el('button', { className: 'btn btn-secondary btn-sm', onClick: () => handleGenerateReport(jobId) }, ['Regenerate']),
      ]),
    ]),
    el('div', { className: 'report-preview', innerHTML: html }),
  ]);
  setApp(wrapper);
}

function renderReportPreview(reportHtml) {
  return el('div', { className: 'report-preview', innerHTML: reportHtml });
}

// --------------- Event Handlers ---------------

async function handleExtractFrames(jobId) {
  try {
    await API.request('POST', `/api/jobs/${jobId}/extract-frames`);
    showToast('Extracting frames...', 'info');
    loadJobDetail(jobId);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleStartProcessing(jobId) {
  try {
    await API.startProcessing(jobId);
    showToast('Processing started', 'info');
    loadJobDetail(jobId);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleRegisterVideo(jobId, path) {
  try {
    await API.registerVideo(jobId, path);
    showToast('Video registered', 'success');
    loadJobDetail(jobId);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleUpdateScreenshot(id, data) {
  try {
    await API.updateScreenshot(id, data);
    showToast('Screenshot updated', 'success');
    // Refresh the card in place rather than reloading everything
    const currentHash = window.location.hash;
    const match = currentHash.match(/^#\/jobs\/([^/]+)/);
    if (match) loadScreenshots(match[1]);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleGenerateReport(jobId) {
  try {
    showToast('Generating report...', 'info');
    await API.generateReport(jobId);
    showToast('Report generated', 'success');
    window.location.hash = `#/jobs/${jobId}/report`;
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleAutoTitle(jobId) {
  try {
    const job = await API.autoTitle(jobId);
    showToast(`Title updated to "${job.title}"`, 'success');
    loadJobDetail(jobId);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleDeleteJob(jobId) {
  if (!confirm('Delete this job? This cannot be undone.')) return;
  try {
    await API.deleteJob(jobId);
    showToast('Job deleted', 'success');
    window.location.hash = '#/';
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleSeedDemo() {
  try {
    await API.seedDemo();
    showToast('Demo data seeded', 'success');
    loadJobs();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// --------------- Polling ---------------

let pollingTimer = null;

function startPolling(jobId) {
  stopPolling();
  pollingTimer = setInterval(async () => {
    try {
      const job = await API.getJob(jobId);
      if (job.status !== 'processing') {
        stopPolling();
        showToast(`Processing ${job.status}`, job.status === 'completed' ? 'success' : 'error');
        loadJobDetail(jobId);
      }
    } catch {
      stopPolling();
    }
  }, 3000);
}

function stopPolling() {
  if (pollingTimer) {
    clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

// --------------- Router ---------------

function route() {
  stopPolling();
  const hash = window.location.hash || '#/';

  const reportMatch = hash.match(/^#\/jobs\/([^/]+)\/report$/);
  if (reportMatch) {
    loadReport(reportMatch[1]);
    updateNavActive('jobs');
    return;
  }

  const jobMatch = hash.match(/^#\/jobs\/([^/]+)$/);
  if (jobMatch) {
    loadJobDetail(jobMatch[1]);
    updateNavActive('jobs');
    return;
  }

  // Default: job list
  loadJobs();
  updateNavActive('jobs');
}

function updateNavActive(section) {
  document.querySelectorAll('.nav-links a').forEach((a) => {
    a.classList.toggle('active', a.dataset.section === section);
  });
}

// --------------- Init ---------------

window.addEventListener('hashchange', route);
window.addEventListener('DOMContentLoaded', () => {
  route();
});
