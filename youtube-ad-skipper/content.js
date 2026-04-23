/**
 * YouTube Ad Auto-Skipper — Content Script
 *
 * Features:
 *  1. Auto-click the skip button as soon as it appears
 *  2. Mute ad audio while waiting for the skip button
 *  3. Speed up unskippable ads to 16x (+ mute)
 *  4. Resilient text-based skip detection (survives class name changes)
 *  5. Handle back-to-back double-ad stacks
 *  6. Close lingering overlay banners and prompts
 *  7. Report skip count to background service worker for badge display
 */

(function () {
  "use strict";

  // ─── Configuration ────────────────────────────────────────────────

  const UNSKIPPABLE_PLAYBACK_RATE = 16;
  const POLL_INTERVAL_MS = 500;

  // ─── Selector lists ───────────────────────────────────────────────

  // Class-based selectors for the skip button (YouTube rotates these)
  const SKIP_SELECTORS = [
    ".ytp-skip-ad-button",
    ".ytp-ad-skip-button",
    ".ytp-ad-skip-button-modern",
    "button.ytp-ad-skip-button",
    ".ytp-ad-skip-button-slot button",
    ".ytp-ad-skip-button-slot",
    'button[id^="skip-button"]',
    ".videoAdUiSkipButton",
    'button[data-id="skip-button"]',
    ".ytp-ad-skip-button-container button",
    ".ytp-ad-skip-button-container",
  ];

  // Overlay / banner close buttons
  const OVERLAY_SELECTORS = [
    ".ytp-ad-overlay-close-button",
    ".ytp-ad-overlay-close-container",
    'button[aria-label="Close"]',
    ".iv-close-button",
  ];

  // Selectors that indicate an ad is currently playing
  const AD_INDICATOR_SELECTORS = [
    ".ad-showing",
    ".ad-interrupting",
    ".ytp-ad-player-overlay",
    ".ytp-ad-player-overlay-instream-info",
  ];

  const COMBINED_SKIP_SELECTOR = SKIP_SELECTORS.join(", ");
  const COMBINED_OVERLAY_SELECTOR = OVERLAY_SELECTORS.join(", ");

  // Text patterns for skip buttons across languages
  // YouTube translates "Skip Ad" / "Skip Ads" into many languages
  const SKIP_TEXT_PATTERNS = [
    /skip\s*ad/i,
    /skip\s*ads/i,
    /skip$/i,                    // just "Skip" at the end of text
    /passer/i,                   // French
    /überspringen/i,             // German
    /saltar/i,                   // Spanish
    /pular/i,                    // Portuguese
    /スキップ/,                   // Japanese
    /광고 건너뛰기/,              // Korean
    /跳过/,                       // Chinese (Simplified)
    /跳過/,                       // Chinese (Traditional)
    /пропустить/i,               // Russian
    /atla/i,                     // Turkish
    /lewati/i,                   // Indonesian
    /ข้าม/,                      // Thai
    /تخطي/,                      // Arabic
    /छोड़ें/,                      // Hindi
  ];

  // ─── State ────────────────────────────────────────────────────────

  let savedVolume = null;        // original volume before we muted
  let savedPlaybackRate = null;  // original playback rate
  let adActive = false;          // are we currently handling an ad?

  // ─── Helpers ──────────────────────────────────────────────────────

  /** Get the main <video> element inside the YouTube player. */
  function getVideoElement() {
    return document.querySelector("video.html5-main-video") ||
           document.querySelector("#movie_player video") ||
           document.querySelector("video");
  }

  /** Check whether an ad is currently playing by looking for ad indicators. */
  function isAdPlaying() {
    // The .ad-showing class on the player container is the most reliable signal
    const player = document.querySelector("#movie_player, .html5-video-player");
    if (player && player.classList.contains("ad-showing")) return true;

    // Fallback: check for any ad indicator element
    return !!document.querySelector(AD_INDICATOR_SELECTORS.join(", "));
  }

  /** Check if a button looks like a skip button based on its text content. */
  function looksLikeSkipButton(el) {
    const text = (el.textContent || "").trim();
    if (!text) return false;
    return SKIP_TEXT_PATTERNS.some((pat) => pat.test(text));
  }

  /**
   * Check if a DOM element is visible and clickable.
   *
   * NOTE: We intentionally do NOT use `offsetParent` here. YouTube's
   * skip button lives inside a fixed-position overlay, and
   * `offsetParent` returns null for elements inside `position: fixed`
   * containers — causing every skip button to look "invisible."
   * Instead we use `getBoundingClientRect()` which works regardless
   * of positioning mode.
   */
  function isClickable(el) {
    if (!el) return false;
    if (el.disabled) return false;

    const style = window.getComputedStyle(el);
    if (style.display === "none") return false;
    if (style.visibility === "hidden") return false;
    if (style.opacity === "0") return false;

    // Check that the element has real dimensions on screen
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  /** Notify the background service worker that we skipped an ad. */
  function notifySkipped() {
    try {
      chrome.runtime.sendMessage({ type: "ad_skipped" });
    } catch {
      // Extension context may be invalidated after update; ignore
    }
  }

  // ─── Core actions ─────────────────────────────────────────────────

  /**
   * Mute the video and save the original volume so we can restore it.
   * Only acts if we haven't already saved the volume.
   */
  function muteAd() {
    const video = getVideoElement();
    if (!video) return;

    if (savedVolume === null) {
      savedVolume = video.volume;
      video.muted = true;
    }
  }

  /**
   * Speed up an unskippable ad and mute it.
   * Only acts if we haven't already changed the playback rate.
   */
  function speedUpAd() {
    const video = getVideoElement();
    if (!video) return;

    if (savedPlaybackRate === null) {
      savedPlaybackRate = video.playbackRate;
      video.playbackRate = UNSKIPPABLE_PLAYBACK_RATE;
    }
    muteAd();
  }

  /** Restore volume and playback rate to pre-ad values. */
  function restorePlayback() {
    const video = getVideoElement();
    if (!video) return;

    if (savedVolume !== null) {
      video.volume = savedVolume;
      video.muted = false;
      savedVolume = null;
    }

    if (savedPlaybackRate !== null) {
      video.playbackRate = savedPlaybackRate;
      savedPlaybackRate = null;
    }
  }

  /**
   * Aggressively click an element using multiple strategies.
   * Some YouTube buttons swallow simple .click() calls, so we also
   * dispatch a full pointer-event sequence.
   */
  function forceClick(el) {
    // Strategy 1: plain click
    el.click();

    // Strategy 2: dispatch a real mouse-event sequence
    for (const type of ["pointerdown", "mousedown", "pointerup", "mouseup", "click"]) {
      el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }));
    }
  }

  /**
   * Try to find and click any visible skip button.
   * Uses both class-based selectors and text-based heuristics.
   * Returns true if a button was clicked.
   */
  function trySkip() {
    // 1. Try class-based selectors first (fastest)
    const classButtons = document.querySelectorAll(COMBINED_SKIP_SELECTOR);
    for (const btn of classButtons) {
      if (isClickable(btn)) {
        forceClick(btn);
        console.log("[Ad Skipper] Skipped ad (class selector):", btn.className || btn.id);
        notifySkipped();
        return true;
      }
      // Some containers aren't clickable themselves but have a
      // clickable child button — try those too
      const inner = btn.querySelector("button, [role='button'], span[role='button']");
      if (inner && isClickable(inner)) {
        forceClick(inner);
        console.log("[Ad Skipper] Skipped ad (inner button):", inner.className || inner.id);
        notifySkipped();
        return true;
      }
    }

    // 2. Fallback: text-based heuristic — find any visible button whose
    //    text matches known skip patterns. This survives YouTube renaming
    //    CSS classes.
    const allButtons = document.querySelectorAll(
      "button, [role='button'], .ytp-ad-skip-button-slot *, .ytp-ad-module *"
    );
    for (const btn of allButtons) {
      if (isClickable(btn) && looksLikeSkipButton(btn)) {
        forceClick(btn);
        console.log("[Ad Skipper] Skipped ad (text heuristic):", btn.textContent.trim());
        notifySkipped();
        return true;
      }
    }

    return false;
  }

  /**
   * Close overlay ads and lingering banners (the semi-transparent
   * boxes that sit on top of the video).
   */
  function closeOverlays() {
    const buttons = document.querySelectorAll(COMBINED_OVERLAY_SELECTOR);
    for (const btn of buttons) {
      if (isClickable(btn)) {
        forceClick(btn);
        console.log("[Ad Skipper] Closed overlay ad.");
        notifySkipped();
      }
    }
  }

  /**
   * Checks if the current ad is unskippable (no skip button present)
   * by looking for the ad countdown timer without a skip button.
   */
  function isUnskippableAd() {
    // If a skip button already exists (even if not yet clickable), it's skippable
    const skipExists = document.querySelector(COMBINED_SKIP_SELECTOR);
    if (skipExists) return false;

    // Check for text-based skip buttons too
    const allButtons = document.querySelectorAll(
      "button, [role='button'], .ytp-ad-skip-button-slot *"
    );
    for (const btn of allButtons) {
      if (looksLikeSkipButton(btn)) return false;
    }

    // If we're in an ad but no skip button exists anywhere, it's unskippable
    return true;
  }

  // ─── Main loop ────────────────────────────────────────────────────

  function tick() {
    const adPlaying = isAdPlaying();

    if (adPlaying) {
      if (!adActive) {
        adActive = true;
        console.log("[Ad Skipper] Ad detected.");
      }

      // Try to skip first
      const skipped = trySkip();

      if (!skipped) {
        // Ad is playing but we couldn't skip yet
        if (isUnskippableAd()) {
          // Unskippable ad: speed it up
          speedUpAd();
        } else {
          // Skippable ad, but button not ready yet: just mute
          muteAd();
        }
      }

      // Always try closing overlay ads
      closeOverlays();

      if (skipped) {
        // After skipping, give YouTube a moment then check for
        // a second ad in a double-ad stack
        setTimeout(() => {
          if (isAdPlaying()) {
            console.log("[Ad Skipper] Double-ad detected, handling second ad.");
            tick();
          } else {
            // Ad is truly over — restore playback settings
            restorePlayback();
            adActive = false;
          }
        }, 500);
        return;
      }
    } else if (adActive) {
      // Ad just ended (either naturally or we skipped it)
      restorePlayback();
      adActive = false;
      console.log("[Ad Skipper] Ad ended, playback restored.");
    }
  }

  // ─── Observers & intervals ────────────────────────────────────────

  // MutationObserver for fast detection of new skip buttons
  const observer = new MutationObserver(() => {
    if (isAdPlaying()) {
      tick();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class", "style", "disabled"],
  });

  // Fallback interval for anything the observer misses
  setInterval(tick, POLL_INTERVAL_MS);

  // Immediate first check
  tick();

  // ─── Handle YouTube SPA navigation ────────────────────────────────
  // YouTube is a single-page app — "navigating" to a new video doesn't
  // reload the page, so our script stays alive. We watch for URL changes
  // to reset state cleanly.

  let lastUrl = location.href;
  const urlObserver = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      restorePlayback();
      adActive = false;
    }
  });

  urlObserver.observe(document.querySelector("head > title") || document.head, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  console.log("[Ad Skipper] YouTube Ad Auto-Skipper v2.1 is active.");
})();
