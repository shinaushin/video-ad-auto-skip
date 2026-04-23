/**
 * Background Service Worker — YouTube Ad Auto-Skipper
 *
 * Keeps a per-tab skip counter and displays it as a badge on the
 * extension icon.  Resets when the tab navigates to a new page.
 */

// Per-tab counters: tabId → number of ads skipped
const skipCounts = new Map();

// --- Badge helpers ---

function updateBadge(tabId) {
  const count = skipCounts.get(tabId) || 0;
  const text = count > 0 ? String(count) : "";

  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color: "#CC0000", tabId });
}

// --- Message handling ---

chrome.runtime.onMessage.addListener((message, sender) => {
  if (!sender.tab) return;
  const tabId = sender.tab.id;

  switch (message.type) {
    case "ad_skipped":
      skipCounts.set(tabId, (skipCounts.get(tabId) || 0) + 1);
      updateBadge(tabId);
      break;

    case "reset_count":
      skipCounts.set(tabId, 0);
      updateBadge(tabId);
      break;
  }
});

// Reset counter when a tab navigates to a new URL
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === "loading") {
    skipCounts.set(tabId, 0);
    updateBadge(tabId);
  }
});

// Clean up when a tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  skipCounts.delete(tabId);
});
