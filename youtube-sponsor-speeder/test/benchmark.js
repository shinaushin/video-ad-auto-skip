#!/usr/bin/env node

/**
 * Sponsor Detection Benchmark
 *
 * Compares our transcript-based sponsor detection against SponsorBlock's
 * community-maintained ground truth.
 *
 * Modes:
 *
 *   1. DATABASE MODE (recommended for large-scale testing):
 *      Downloads SponsorBlock's full sponsorTimes.csv database dump,
 *      samples N random videos with high-confidence sponsor segments,
 *      fetches their YouTube captions, and benchmarks our detection.
 *
 *      node benchmark.js --db                  # sample 50 videos
 *      node benchmark.js --db --sample 200     # sample 200 videos
 *      node benchmark.js --db --concurrency 5  # parallel caption fetches
 *
 *   2. API MODE (for testing specific videos):
 *      Fetches ground truth per-video from the SponsorBlock API.
 *
 *      node benchmark.js VIDEO_ID [VIDEO_ID ...]
 *
 * Requirements:
 *   Node.js 18+ (zero npm dependencies)
 */

const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { createGunzip } = require("zlib");

// ─── Minimal XML parser ────────────────────────────────────────

function parseTimedTextXML(xml) {
  const cues = [];
  const regex = /<text\s+start="([^"]+)"\s+dur="([^"]*)"[^>]*>([\s\S]*?)<\/text>/g;
  let match;
  while ((match = regex.exec(xml)) !== null) {
    cues.push({
      start: parseFloat(match[1]),
      dur: parseFloat(match[2] || "2"),
      text: match[3]
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&#39;/g, "'")
        .replace(/&quot;/g, '"')
        .replace(/\n/g, " ")
        .trim(),
    });
  }
  return cues;
}

// ─── Detection logic (mirrored from content.js) ────────────────

const STRONG_PATTERNS = [
  /this (?:video|segment|portion) is (?:brought to you|sponsored|made possible) by/i,
  /(?:sponsored|presented) by/i,
  /today'?s sponsor/i,
  /thanks to .{1,40} for sponsoring/i,
  /a (?:huge|big|special) thanks? to/i,
  /brought to you by/i,
  /use (?:my |our )?(?:code|link)/i,
  /use code .{1,20} (?:at|for) checkout/i,
  /go to .{1,40}\.com/i,
  /head (?:on )?over to .{1,40}\.com/i,
  /visit .{1,40}\.com/i,
  /check (?:them )?out at/i,
  /click (?:the|my) link/i,
  /link (?:is )?in (?:the )?description/i,
  /first \d+ (?:people|users|customers|subscribers)/i,
  /free trial/i,
  /percent off/i,
  /% off/i,
  /discount code/i,
  /promo code/i,
  /coupon code/i,
];

const WEAK_PATTERNS = [
  /sign up/i,
  /download the app/i,
  /available (?:now )?at/i,
  /money back guarantee/i,
  /limited time/i,
  /exclusive (?:deal|offer)/i,
  /subscribe/i,
  /premium/i,
];

const PADDING_BEFORE = 1.5;
const PADDING_AFTER = 2.0;
const MERGE_GAP_SEC = 8;
const MIN_KEYWORD_HITS = 2;

function scoreCue(text) {
  let score = 0;
  for (const pat of STRONG_PATTERNS) if (pat.test(text)) score += 3;
  for (const pat of WEAK_PATTERNS) if (pat.test(text)) score += 1;
  return score;
}

function detectSponsorSegments(cues) {
  if (!cues.length) return [];
  const scored = cues.map((cue) => ({
    ...cue,
    end: cue.start + cue.dur,
    score: scoreCue(cue.text),
  }));
  const hits = scored.filter((c) => c.score > 0);
  if (!hits.length) return [];

  const rawSegments = [];
  let cluster = [hits[0]];
  for (let i = 1; i < hits.length; i++) {
    const prev = cluster[cluster.length - 1];
    const curr = hits[i];
    if (curr.start - prev.end <= MERGE_GAP_SEC) {
      cluster.push(curr);
    } else {
      rawSegments.push(cluster);
      cluster = [curr];
    }
  }
  rawSegments.push(cluster);

  const segments = [];
  for (const cl of rawSegments) {
    const totalScore = cl.reduce((sum, c) => sum + c.score, 0);
    if (totalScore < MIN_KEYWORD_HITS * 3) continue;
    const start = Math.max(0, cl[0].start - PADDING_BEFORE);
    const end = cl[cl.length - 1].end + PADDING_AFTER;
    segments.push({ start, end });
  }
  return segments;
}

// ─── HTTP helpers ──────────────────────────────────────────────

function fetch(url, opts = {}) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith("https") ? https : http;
    const req = mod.get(url, { headers: { "User-Agent": "SponsorSpeedBenchmark/1.0" }, ...opts }, (res) => {
      // Follow redirects
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return fetch(res.headers.location, opts).then(resolve, reject);
      }
      resolve(res);
    });
    req.on("error", reject);
  });
}

function fetchText(url) {
  return new Promise(async (resolve, reject) => {
    try {
      const res = await fetch(url);
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => resolve(data));
      res.on("error", reject);
    } catch (e) {
      reject(e);
    }
  });
}

function fetchJSON(url) {
  return new Promise(async (resolve, reject) => {
    try {
      const res = await fetch(url);
      if (res.statusCode === 404) return resolve(null);
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(e); }
      });
      res.on("error", reject);
    } catch (e) {
      reject(e);
    }
  });
}

// ─── YouTube caption fetching ──────────────────────────────────

async function getCaptionsForVideo(videoId) {
  const html = await fetchText(`https://www.youtube.com/watch?v=${videoId}`);
  const match = html.match(/"captionTracks":\s*(\[.*?\])/);
  if (!match) return null;

  let tracks;
  try { tracks = JSON.parse(match[1]); } catch { return null; }
  if (!tracks || !tracks.length) return null;

  const english = tracks.find((t) => t.languageCode?.startsWith("en"));
  const track = english || tracks[0];
  if (!track?.baseUrl) return null;

  const xml = await fetchText(track.baseUrl);
  return parseTimedTextXML(xml);
}

// ─── SponsorBlock API (single-video mode) ──────────────────────

async function getSponsorBlockSegments(videoId) {
  const url = `https://sponsor.ajay.app/api/skipSegments?videoID=${videoId}&categories=${encodeURIComponent('["sponsor"]')}`;
  const data = await fetchJSON(url);
  if (!data || !Array.isArray(data)) return [];
  return data
    .filter((seg) => seg.category === "sponsor" && seg.actionType === "skip")
    .map((seg) => ({ start: seg.segment[0], end: seg.segment[1] }));
}

// ═══════════════════════════════════════════════════════════════
//  DATABASE MODE — download & parse SponsorBlock's full CSV dump
// ═══════════════════════════════════════════════════════════════

const DB_CACHE_DIR = path.join(__dirname, ".cache");
const DB_CACHE_FILE = path.join(DB_CACHE_DIR, "sponsorTimes.csv");

// SponsorBlock database mirror (updates every ~30 min)
const DB_MIRRORS = [
  "https://sb.ltn.fi/database/sponsorTimes.csv",
  "https://mirror.sb.mchang.xyz/sponsorTimes.csv",
  "https://sponsor.ajay.app/database/sponsorTimes.csv",
];

/**
 * Download the SponsorBlock sponsorTimes.csv if not cached.
 * The file is large (~2-4 GB), so we stream it.
 *
 * Returns the path to the local CSV file.
 */
async function ensureDatabase() {
  if (!fs.existsSync(DB_CACHE_DIR)) fs.mkdirSync(DB_CACHE_DIR, { recursive: true });

  if (fs.existsSync(DB_CACHE_FILE)) {
    const stats = fs.statSync(DB_CACHE_FILE);
    const ageHours = (Date.now() - stats.mtimeMs) / 3600000;
    const sizeMB = (stats.size / 1e6).toFixed(0);
    if (stats.size > 1e6) {
      console.log(`Using cached database (${sizeMB} MB, ${ageHours.toFixed(1)}h old)`);
      if (ageHours > 168) {
        console.log("  Hint: database is over a week old. Delete .cache/ to re-download.\n");
      }
      return DB_CACHE_FILE;
    }
  }

  console.log("Downloading SponsorBlock database...");
  console.log("(This is a large file, ~2-4 GB. It will be cached for future runs.)\n");

  for (const mirrorUrl of DB_MIRRORS) {
    try {
      console.log(`Trying ${mirrorUrl}...`);
      await downloadFile(mirrorUrl, DB_CACHE_FILE);
      const sizeMB = (fs.statSync(DB_CACHE_FILE).size / 1e6).toFixed(0);
      console.log(`Download complete (${sizeMB} MB)\n`);
      return DB_CACHE_FILE;
    } catch (e) {
      console.log(`  Failed: ${e.message}`);
    }
  }

  throw new Error(
    "Could not download the SponsorBlock database from any mirror.\n" +
    "You can download it manually from https://sb.ltn.fi/database/\n" +
    `and place it at: ${DB_CACHE_FILE}`
  );
}

function downloadFile(url, dest) {
  return new Promise(async (resolve, reject) => {
    try {
      const res = await fetch(url);
      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode}`));
      }

      const total = parseInt(res.headers["content-length"] || "0", 10);
      let downloaded = 0;
      let lastPct = -1;

      const file = fs.createWriteStream(dest);
      res.pipe(file);

      res.on("data", (chunk) => {
        downloaded += chunk.length;
        if (total > 0) {
          const pct = Math.floor((downloaded / total) * 100);
          if (pct !== lastPct && pct % 5 === 0) {
            process.stdout.write(`  ${pct}% (${(downloaded / 1e6).toFixed(0)} MB)\r`);
            lastPct = pct;
          }
        }
      });

      file.on("finish", () => {
        process.stdout.write("\n");
        file.close(resolve);
      });
      file.on("error", (e) => {
        fs.unlinkSync(dest);
        reject(e);
      });
    } catch (e) {
      reject(e);
    }
  });
}

/**
 * Parse the sponsorTimes.csv and extract high-confidence sponsor
 * segments grouped by videoID.
 *
 * SponsorBlock CSV columns:
 *   videoID, startTime, endTime, votes, locked, incorrectVotes,
 *   UUID, userID, timeSubmitted, views, category, actionType,
 *   service, videoDuration, hidden, reputation, shadowHidden,
 *   hashedVideoID, userAgent, description
 *
 * We filter for:
 *   - category = "sponsor"
 *   - votes >= 1 (positive community validation)
 *   - hidden = 0, shadowHidden = 0
 *   - videoDuration > 60 (skip shorts)
 */
async function parseDatabase(csvPath, sampleSize = 50) {
  console.log("Parsing database for high-confidence sponsor segments...");

  // Map: videoID → [{start, end, votes}]
  const videoSegments = new Map();

  const fileStream = fs.createReadStream(csvPath, { encoding: "utf-8" });
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  let headers = null;
  let lineCount = 0;
  let matchCount = 0;

  for await (const line of rl) {
    lineCount++;

    // Parse header row
    if (!headers) {
      headers = parseCSVRow(line);
      continue;
    }

    // Show progress every 500k lines
    if (lineCount % 500000 === 0) {
      process.stdout.write(`  Scanned ${(lineCount / 1e6).toFixed(1)}M rows, found ${videoSegments.size} qualifying videos...\r`);
    }

    const cols = parseCSVRow(line);
    if (cols.length < headers.length) continue;

    const row = {};
    for (let i = 0; i < headers.length; i++) {
      row[headers[i]] = cols[i];
    }

    // Filter criteria
    if (row.category !== "sponsor") continue;
    if (row.hidden === "1" || row.shadowHidden === "1") continue;
    if (parseFloat(row.votes || "0") < 1) continue;
    if (parseFloat(row.videoDuration || "0") < 60) continue;
    if (row.service && row.service !== "YouTube") continue;

    const vid = row.videoID;
    if (!vid || vid.length !== 11) continue;

    if (!videoSegments.has(vid)) videoSegments.set(vid, []);
    videoSegments.get(vid).push({
      start: parseFloat(row.startTime),
      end: parseFloat(row.endTime),
      votes: parseInt(row.votes || "0", 10),
    });

    matchCount++;
  }

  process.stdout.write("\n");
  console.log(`Scanned ${lineCount.toLocaleString()} rows total`);
  console.log(`Found ${matchCount.toLocaleString()} qualifying sponsor segments across ${videoSegments.size.toLocaleString()} videos`);

  // Filter to videos with at least one segment with votes >= 3
  // (higher confidence) and merge overlapping segments
  const highConfidence = [];
  for (const [vid, segs] of videoSegments) {
    const hasStrong = segs.some((s) => s.votes >= 3);
    if (!hasStrong) continue;

    // Merge overlapping/adjacent segments
    const sorted = segs.sort((a, b) => a.start - b.start);
    const merged = [{ ...sorted[0] }];
    for (let i = 1; i < sorted.length; i++) {
      const prev = merged[merged.length - 1];
      const curr = sorted[i];
      if (curr.start <= prev.end + 2) {
        prev.end = Math.max(prev.end, curr.end);
        prev.votes = Math.max(prev.votes, curr.votes);
      } else {
        merged.push({ ...curr });
      }
    }

    highConfidence.push({ videoId: vid, segments: merged });
  }

  console.log(`High-confidence videos (votes >= 3): ${highConfidence.length.toLocaleString()}`);

  // Random sample
  const shuffled = highConfidence.sort(() => Math.random() - 0.5);
  const sample = shuffled.slice(0, sampleSize);
  console.log(`Sampled ${sample.length} videos for benchmarking\n`);

  return sample;
}

/**
 * Naive CSV row parser that handles quoted fields.
 */
function parseCSVRow(line) {
  const cols = [];
  let i = 0;
  while (i < line.length) {
    if (line[i] === ",") {
      cols.push("");
      i++;
    } else if (line[i] === '"') {
      let val = "";
      i++; // skip opening quote
      while (i < line.length) {
        if (line[i] === '"' && line[i + 1] === '"') {
          val += '"';
          i += 2;
        } else if (line[i] === '"') {
          i++; // skip closing quote
          break;
        } else {
          val += line[i];
          i++;
        }
      }
      cols.push(val);
      if (line[i] === ",") i++; // skip comma after quoted field
    } else {
      let val = "";
      while (i < line.length && line[i] !== ",") {
        val += line[i];
        i++;
      }
      cols.push(val);
      if (line[i] === ",") i++;
    }
  }
  return cols;
}

// ─── Metrics ───────────────────────────────────────────────────

function computeMetrics(detected, groundTruth) {
  if (groundTruth.length === 0 && detected.length === 0) {
    return { precision: 1, recall: 1, iou: 1, trueNegative: true, details: [] };
  }
  if (groundTruth.length === 0 && detected.length > 0) {
    return {
      precision: 0, recall: 1, iou: 0, falsePositives: detected.length,
      details: detected.map((d) => ({ type: "false_positive", detected: d })),
    };
  }
  if (detected.length === 0 && groundTruth.length > 0) {
    return {
      precision: 1, recall: 0, iou: 0, missedSegments: groundTruth.length,
      details: groundTruth.map((gt) => ({ type: "missed", groundTruth: gt })),
    };
  }

  let detectedTotalTime = 0, gtTotalTime = 0;
  for (const gt of groundTruth) gtTotalTime += gt.end - gt.start;
  for (const d of detected) detectedTotalTime += d.end - d.start;

  // Sweep-line intersection
  const events = [];
  for (const gt of groundTruth) {
    events.push({ time: gt.start, type: "gs" });
    events.push({ time: gt.end, type: "ge" });
  }
  for (const d of detected) {
    events.push({ time: d.start, type: "ds" });
    events.push({ time: d.end, type: "de" });
  }
  events.sort((a, b) => a.time - b.time);

  let gtActive = 0, detActive = 0, prevTime = 0, intersectionTime = 0;
  for (const ev of events) {
    const dt = ev.time - prevTime;
    if (dt > 0 && gtActive > 0 && detActive > 0) intersectionTime += dt;
    prevTime = ev.time;
    if (ev.type === "gs") gtActive++;
    else if (ev.type === "ge") gtActive--;
    else if (ev.type === "ds") detActive++;
    else detActive--;
  }

  const unionTime = detectedTotalTime + gtTotalTime - intersectionTime;
  const precision = detectedTotalTime > 0 ? intersectionTime / detectedTotalTime : 0;
  const recall = gtTotalTime > 0 ? intersectionTime / gtTotalTime : 0;
  const iou = unionTime > 0 ? intersectionTime / unionTime : 0;

  const details = groundTruth.map((gt) => {
    let overlap = 0;
    for (const d of detected) {
      const s = Math.max(gt.start, d.start), e = Math.min(gt.end, d.end);
      if (e > s) overlap += e - s;
    }
    return { groundTruth: gt, coverage: (gt.end - gt.start) > 0 ? overlap / (gt.end - gt.start) : 0 };
  });

  return { precision, recall, iou, details };
}

// ─── Formatting ────────────────────────────────────────────────

function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
function pct(n) { return (n * 100).toFixed(1) + "%"; }
function segStr(seg) {
  return `${fmt(seg.start)} → ${fmt(seg.end)} (${(seg.end - seg.start).toFixed(0)}s)`;
}

// ─── Benchmark a single video ──────────────────────────────────

async function benchmarkVideo(videoId, groundTruth, verbose = true) {
  if (verbose) {
    const header = `── Video: ${videoId} `;
    console.log(`\n${header}${"─".repeat(Math.max(0, 60 - header.length))}`);
    console.log(`  Ground truth: ${groundTruth.length} segment(s)`);
    if (groundTruth.length > 0) {
      console.log("    " + groundTruth.map(segStr).join("\n    "));
    }
  }

  let cues;
  try {
    cues = await getCaptionsForVideo(videoId);
  } catch (e) {
    if (verbose) console.log(`  Captions: error — ${e.message}`);
    return { videoId, groundTruth, detected: [], metrics: null, error: e.message };
  }

  if (!cues || !cues.length) {
    if (verbose) console.log("  Captions: not available");
    return { videoId, groundTruth, detected: [], metrics: null, noCaptions: true };
  }

  if (verbose) console.log(`  Captions: ${cues.length} cues`);

  const detected = detectSponsorSegments(cues);
  if (verbose) {
    console.log(`  Detected: ${detected.length} segment(s)`);
    if (detected.length > 0) console.log("    " + detected.map(segStr).join("\n    "));
  }

  const metrics = computeMetrics(detected, groundTruth);
  if (verbose) {
    console.log(`  Precision: ${pct(metrics.precision)}  Recall: ${pct(metrics.recall)}  IoU: ${pct(metrics.iou)}`);
    if (metrics.details) {
      for (const d of metrics.details) {
        if (d.type === "false_positive") console.log(`  ⚠ False positive: ${segStr(d.detected)}`);
        else if (d.type === "missed") console.log(`  ✗ Missed: ${segStr(d.groundTruth)}`);
        else if (d.coverage !== undefined) {
          const icon = d.coverage >= 0.5 ? "✓" : "✗";
          console.log(`  ${icon} GT ${segStr(d.groundTruth)} — ${pct(d.coverage)} covered`);
        }
      }
    }
  }

  return { videoId, groundTruth, detected, metrics };
}

// ─── Parallel execution with concurrency limit ────────────────

async function parallelMap(items, fn, concurrency) {
  const results = [];
  let idx = 0;

  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      results[i] = await fn(items[i], i);
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

// ─── Main ──────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const dbMode = args.includes("--db");
  const verbose = !args.includes("--quiet");

  let sampleSize = 50;
  const sampleIdx = args.indexOf("--sample");
  if (sampleIdx !== -1) sampleSize = parseInt(args[sampleIdx + 1]) || 50;

  let concurrency = 3;
  const concIdx = args.indexOf("--concurrency");
  if (concIdx !== -1) concurrency = parseInt(args[concIdx + 1]) || 3;

  console.log("YouTube Sponsor Speeder — Detection Benchmark");
  console.log("═".repeat(50));

  let testCases; // [{videoId, segments: [{start, end}]}]

  if (dbMode) {
    // ── Database mode: download & sample from full SponsorBlock DB ──
    const csvPath = await ensureDatabase();
    testCases = await parseDatabase(csvPath, sampleSize);
  } else {
    // ── API mode: specific video IDs ──
    const videoIds = args.filter((a) => !a.startsWith("--"));

    if (videoIds.length === 0) {
      console.log(`
Usage:

  DATABASE MODE (large-scale, recommended):
    node benchmark.js --db                   # sample 50 random videos
    node benchmark.js --db --sample 200      # sample 200 videos
    node benchmark.js --db --sample 500 --concurrency 5 --quiet

  API MODE (specific videos):
    node benchmark.js VIDEO_ID [VIDEO_ID ...]

  Options:
    --db            Download SponsorBlock's full database (~2-4 GB, cached)
    --sample N      Number of videos to sample in DB mode (default: 50)
    --concurrency N Parallel YouTube caption fetches (default: 3)
    --quiet         Less verbose per-video output

  The database is cached in .cache/ and reused on subsequent runs.
  Delete .cache/ to force a fresh download.
`);
      process.exit(0);
    }

    // Fetch ground truth from API for each video
    testCases = [];
    for (const vid of videoIds) {
      try {
        const segments = await getSponsorBlockSegments(vid);
        testCases.push({ videoId: vid, segments });
      } catch (e) {
        console.log(`  Warning: Could not fetch SponsorBlock data for ${vid}: ${e.message}`);
      }
    }
  }

  console.log(`Testing ${testCases.length} video(s) with concurrency ${concurrency}...\n`);

  // Run benchmarks
  let completed = 0;
  const results = await parallelMap(
    testCases,
    async (tc, i) => {
      // Rate limit: small delay between requests to be polite
      await new Promise((r) => setTimeout(r, i * 300));
      const result = await benchmarkVideo(tc.videoId, tc.segments, verbose);
      completed++;
      if (!verbose) {
        const status = result.noCaptions ? "no-captions" :
                       result.error ? "error" :
                       `P:${pct(result.metrics.precision)} R:${pct(result.metrics.recall)}`;
        process.stdout.write(`  [${completed}/${testCases.length}] ${tc.videoId} — ${status}\n`);
      }
      return result;
    },
    concurrency
  );

  // ─── Aggregate summary ──────────────────────────────────────

  console.log("\n" + "═".repeat(50));
  console.log("AGGREGATE RESULTS");
  console.log("═".repeat(50));

  const successful = results.filter((r) => r.metrics);
  const withSegments = successful.filter((r) => !r.metrics.trueNegative);
  const noCaptions = results.filter((r) => r.noCaptions);
  const errors = results.filter((r) => r.error);
  const trueNeg = successful.filter((r) => r.metrics.trueNegative);

  console.log(`Total videos tested:     ${results.length}`);
  console.log(`  Successful benchmarks: ${successful.length}`);
  console.log(`  With sponsor segments: ${withSegments.length}`);
  console.log(`  True negatives:        ${trueNeg.length}`);
  console.log(`  No captions available: ${noCaptions.length}`);
  console.log(`  Errors:                ${errors.length}`);

  if (withSegments.length > 0) {
    const avgP = withSegments.reduce((s, r) => s + r.metrics.precision, 0) / withSegments.length;
    const avgR = withSegments.reduce((s, r) => s + r.metrics.recall, 0) / withSegments.length;
    const avgIoU = withSegments.reduce((s, r) => s + r.metrics.iou, 0) / withSegments.length;

    // Also compute segment-level stats
    let totalGT = 0, coveredGT = 0, totalFP = 0;
    for (const r of withSegments) {
      if (!r.metrics.details) continue;
      for (const d of r.metrics.details) {
        if (d.type === "false_positive") totalFP++;
        else if (d.type === "missed") totalGT++;
        else if (d.coverage !== undefined) {
          totalGT++;
          if (d.coverage >= 0.5) coveredGT++;
        }
      }
    }

    console.log("\n── Detection Accuracy ──");
    console.log(`Average Precision:  ${pct(avgP)}`);
    console.log(`Average Recall:     ${pct(avgR)}`);
    console.log(`Average IoU:        ${pct(avgIoU)}`);
    console.log(`\nSegment-level:`);
    console.log(`  Ground truth segments:  ${totalGT}`);
    console.log(`  Correctly covered (≥50%): ${coveredGT} (${totalGT > 0 ? pct(coveredGT / totalGT) : "N/A"})`);
    console.log(`  False positives:        ${totalFP}`);

    console.log("\n── Interpretation ──");
    console.log("  Precision = of the time we flagged, how much was actually a sponsor");
    console.log("  Recall    = of the real sponsor time, how much did we catch");
    console.log("  IoU       = overall overlap quality (higher is better)");

    if (avgR < 0.3) {
      console.log("\n⚠ Low recall: many sponsor segments being missed.");
      console.log("  → Add more keyword patterns or lower MIN_KEYWORD_HITS");
    } else if (avgR < 0.5) {
      console.log("\n⚠ Moderate recall: some sponsors being missed.");
    }
    if (avgP < 0.3) {
      console.log("\n⚠ Low precision: many false positives.");
      console.log("  → Raise MIN_KEYWORD_HITS or remove noisy WEAK_PATTERNS");
    } else if (avgP < 0.5) {
      console.log("\n⚠ Moderate precision: some false positives.");
    }
    if (avgIoU >= 0.4) {
      console.log("\n✓ Decent IoU — detection is roughly aligned with ground truth.");
    }
  }

  // Write detailed results to JSON
  const reportPath = path.join(__dirname, "benchmark-results.json");
  const report = {
    timestamp: new Date().toISOString(),
    mode: dbMode ? "database" : "api",
    sampleSize: testCases.length,
    summary: {
      total: results.length,
      successful: successful.length,
      withSegments: withSegments.length,
      noCaptions: noCaptions.length,
      errors: errors.length,
    },
    averages: withSegments.length > 0 ? {
      precision: withSegments.reduce((s, r) => s + r.metrics.precision, 0) / withSegments.length,
      recall: withSegments.reduce((s, r) => s + r.metrics.recall, 0) / withSegments.length,
      iou: withSegments.reduce((s, r) => s + r.metrics.iou, 0) / withSegments.length,
    } : null,
    videos: results.map((r) => ({
      videoId: r.videoId,
      groundTruthCount: r.groundTruth?.length || 0,
      detectedCount: r.detected?.length || 0,
      precision: r.metrics?.precision,
      recall: r.metrics?.recall,
      iou: r.metrics?.iou,
      noCaptions: r.noCaptions || false,
      error: r.error || null,
    })),
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\nDetailed results written to: ${reportPath}`);
  console.log("");
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
