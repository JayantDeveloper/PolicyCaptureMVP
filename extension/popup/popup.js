document.getElementById('btn-open').addEventListener('click', () => {
  chrome.tabs.create({ url: 'http://localhost:8420/recorder' });
  window.close();
});
document.getElementById('btn-dash').addEventListener('click', () => {
  chrome.tabs.create({ url: 'http://localhost:8420' });
  window.close();
});
