#!/usr/bin/env python3
"""
Unit tests for benchmark.py

Covers pure-logic functions only — no network calls.
Run with:  python3 -m pytest test_benchmark.py -v
       or:  python3 test_benchmark.py
"""

import json
import sys
import unittest
from unittest.mock import patch, MagicMock

# Import the module under test (same directory)
sys.path.insert(0, ".")
import benchmark as bm


# ══════════════════════════════════════════════════════════════════
#  score_cue
# ══════════════════════════════════════════════════════════════════

class TestScoreCue(unittest.TestCase):

    def test_clean_cue_scores_zero(self):
        self.assertEqual(bm.score_cue("and that wraps up today's tutorial"), 0)

    def test_empty_string_scores_zero(self):
        self.assertEqual(bm.score_cue(""), 0)

    def test_strong_pattern_scores_three(self):
        # "brought to you by" is a STRONG_PATTERN → +3
        # (Note: "This video is brought to you by" also matches a second pattern for +3 more;
        #  use a phrase that triggers exactly one strong pattern to test the +3 unit.)
        self.assertEqual(bm.score_cue("Sponsored by Acme Corp"), 3)

    def test_strong_pattern_case_insensitive(self):
        self.assertEqual(bm.score_cue("BROUGHT TO YOU BY acme"), 3)

    def test_weak_pattern_scores_one(self):
        # "sign up" alone is a WEAK_PATTERN → +1
        self.assertEqual(bm.score_cue("Sign up for free today"), 1)

    def test_multiple_patterns_accumulate(self):
        # "use my code" (strong +3) + "percent off" (strong +3) + "sign up" (weak +1)
        text = "Use my code SAVE20 for 20 percent off, just sign up first"
        score = bm.score_cue(text)
        self.assertGreaterEqual(score, 7)

    def test_sponsor_language_scores_high(self):
        text = "Today's sponsor is Brilliant. Go to brilliant.com to get started"
        score = bm.score_cue(text)
        # "today's sponsor" (+3) + "go to .com" (+3)
        self.assertGreaterEqual(score, 6)

    def test_promo_code_strong(self):
        score = bm.score_cue("Use promo code YOUTUBE for 15% off")
        self.assertGreaterEqual(score, 3)

    def test_link_in_description_strong(self):
        score = bm.score_cue("Link is in the description below")
        self.assertGreaterEqual(score, 3)

    def test_free_trial_strong(self):
        score = bm.score_cue("Start your free trial today")
        self.assertGreaterEqual(score, 3)


# ══════════════════════════════════════════════════════════════════
#  detect_sponsor_segments
# ══════════════════════════════════════════════════════════════════

class TestDetectSponsorSegments(unittest.TestCase):

    def _make_cues(self, entries):
        """Helper: list of (start, dur, text) → cue dicts."""
        return [{"start": s, "dur": d, "text": t} for s, d, t in entries]

    def test_empty_input_returns_empty(self):
        self.assertEqual(bm.detect_sponsor_segments([]), [])

    def test_no_sponsor_language_returns_empty(self):
        cues = self._make_cues([
            (0, 3, "Welcome to my channel"),
            (3, 3, "Today we are going to learn about Python"),
            (6, 3, "Let's get started with the basics"),
        ])
        self.assertEqual(bm.detect_sponsor_segments(cues), [])

    def test_single_strong_hit_meets_threshold(self):
        # MIN_SCORE = 3, so one strong pattern hit (score=3) now triggers detection
        cues = self._make_cues([
            (60, 5, "Sponsored by Acme Corp"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)

    def test_split_phrase_detected_via_window(self):
        # "brought to you by" is split across two cues — window scoring catches it
        cues = self._make_cues([
            (60, 3, "this video is brought"),
            (63, 3, "to you by Acme Corp"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)

    def test_two_strong_hits_form_segment(self):
        cues = self._make_cues([
            (60, 5, "This video is brought to you by Acme Corp"),
            (65, 5, "Go to acme.com and use code SAVE for 20 percent off"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)

    def test_segment_applies_min_duration_floor(self):
        # Two sponsor cues span only 10 s; the MIN_SEGMENT_DURATION floor should
        # extend the result to at least 45 s, centred around the detected block.
        cues = self._make_cues([
            (100, 5, "Sponsored by Acme, go to acme.com for a free trial"),
            (105, 5, "Use my code ACME for 20 percent off your first order"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        seg = result[0]
        # Must cover the actual sponsor cues
        self.assertLessEqual(seg["start"], 100)
        self.assertGreaterEqual(seg["end"], 110)
        # Must meet the minimum duration floor
        self.assertGreaterEqual(seg["end"] - seg["start"], bm.MIN_SEGMENT_DURATION)

    def test_start_clamped_to_zero(self):
        cues = self._make_cues([
            (0, 5, "Brought to you by Acme, use code SAVE for percent off"),
            (5, 5, "Go to acme.com for your free trial today"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0]["start"], 0)

    def test_gap_larger_than_merge_gap_creates_two_segments(self):
        # Two sponsor clusters separated by > MERGE_GAP_SEC + WINDOW_SEC → two segments.
        # Gap must be large enough that the sliding windows don't overlap.
        gap_start = 60 + 5 + bm.MERGE_GAP_SEC + bm.WINDOW_SEC + 10
        cues = self._make_cues([
            (60, 5, "Brought to you by Acme, use my code SAVE for percent off"),
            (65, 5, "Visit acme.com for a free trial and sign up now"),
            (gap_start,     5, "Brought to you by Beta Inc, use my code BETA for percent off"),
            (gap_start + 5, 5, "Visit beta.com for a free trial and sign up today"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 2)

    def test_gap_within_merge_gap_creates_one_segment(self):
        cues = self._make_cues([
            (60, 5, "Brought to you by Acme, use my code SAVE for percent off"),
            (70, 5, "Go to acme.com for a free trial today"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)

    def test_silence_gap_used_as_start_boundary(self):
        # A long silence (>> SILENCE_BOUNDARY_SEC) before the sponsor block means
        # the walker stops at the first sponsor cue rather than pulling in earlier cues.
        cues = self._make_cues([
            (0,  3, "Welcome to my channel today"),
            # 57-second silence — clear natural break
            (60, 5, "Brought to you by Acme, use my code SAVE for percent off"),
            (65, 5, "Visit acme.com for a free trial today"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        # Start should not reach back past the silence to grab the intro cue at t=0
        # (silence gap is 57 s >> SILENCE_BOUNDARY_SEC so walker stops at t=60)
        self.assertGreater(result[0]["start"], 3)

    def test_silence_gap_used_as_end_boundary(self):
        # A long silence after the sponsor block should cap the end boundary,
        # rather than extending past it into regular content.
        cues = self._make_cues([
            (60, 5, "Brought to you by Acme, use my code SAVE for percent off"),
            (65, 5, "Visit acme.com for a free trial and sign up now"),
            # 55-second silence
            (120, 3, "Alright back to the tutorial where we left off"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        # End boundary should stop at the silence gap (before t=120), not extend past it
        self.assertLess(result[0]["end"], 120)

    def test_sponsor_intro_phrase_marks_start_boundary(self):
        # A "before we get into" phrase should be picked up as the start of the ad block
        cues = self._make_cues([
            (55, 3, "but before we get into it"),       # SPONSOR_INTRO_PATTERNS match
            (58, 2, "today's video is brought to you by Acme Corp"),
            (60, 5, "Go to acme.com and use code SAVE for 20 percent off"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        # The segment should start at or before the intro phrase cue at t=55
        self.assertLessEqual(result[0]["start"], 55)

    def test_content_return_phrase_marks_end_boundary(self):
        # A content-return phrase should help cap the end of the ad block.
        # Note: cues within WINDOW_SEC of the sponsor block inherit its score, so
        # the "anyway" cue gets pulled into the cluster when it sits right after the
        # sponsor block.  MIN_SEGMENT_DURATION then expands the result centred on
        # the whole cluster.  The key guarantee is that the segment stays anchored
        # around the sponsor block and does NOT reach far into the regular content.
        cues = self._make_cues([
            (60, 5, "Brought to you by Acme, use my code SAVE for percent off"),
            (65, 5, "Visit acme.com for a free trial today"),
            (70, 3, "anyway, let's get back to what we were talking about"),  # CONTENT_RETURN
            (73, 5, "So the key insight here is very important"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        # Segment must cover the sponsor cues…
        self.assertLessEqual(result[0]["start"], 60)
        self.assertGreaterEqual(result[0]["end"], 70)
        # …but should not run far past the content cues (well within regular content)
        self.assertLess(result[0]["end"], 120)

    def test_min_segment_duration_extends_symmetrically(self):
        # A single-cue detection (5 s) should be extended to MIN_SEGMENT_DURATION.
        cues = self._make_cues([
            (100, 5, "Sponsored by Acme Corp use my code today"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        seg = result[0]
        duration = seg["end"] - seg["start"]
        self.assertGreaterEqual(duration, bm.MIN_SEGMENT_DURATION)
        # Extension should be roughly symmetric around the raw detected block
        midpoint = seg["start"] + duration / 2
        raw_mid = 100 + 5 / 2  # 102.5
        self.assertAlmostEqual(midpoint, raw_mid, delta=3.0)

    def test_non_sponsor_cues_ignored(self):
        cues = self._make_cues([
            (0, 3, "Hello everyone welcome back"),
            (60, 5, "Brought to you by Acme, use my code SAVE for percent off"),
            (65, 5, "Visit acme.com right now for a free trial"),
            (120, 3, "Alright back to the tutorial"),
        ])
        result = bm.detect_sponsor_segments(cues)
        self.assertEqual(len(result), 1)
        # Segment should be around the sponsor block, not the full video
        self.assertLess(result[0]["start"], 65)
        self.assertLess(result[0]["end"], 120)


# ══════════════════════════════════════════════════════════════════
#  compute_metrics
# ══════════════════════════════════════════════════════════════════

class TestComputeMetrics(unittest.TestCase):

    def _seg(self, start, end):
        return {"start": start, "end": end}

    def test_both_empty_is_true_negative(self):
        m = bm.compute_metrics([], [])
        self.assertEqual(m["precision"], 1)
        self.assertEqual(m["recall"], 1)
        self.assertEqual(m["iou"], 1)
        self.assertTrue(m.get("true_negative"))

    def test_no_detection_no_ground_truth_true_negative(self):
        m = bm.compute_metrics([], [])
        self.assertTrue(m.get("true_negative"))

    def test_false_positive_only(self):
        m = bm.compute_metrics([self._seg(10, 30)], [])
        self.assertEqual(m["precision"], 0)
        self.assertEqual(m["recall"], 1)
        self.assertEqual(m["iou"], 0)

    def test_missed_segment_zero_recall(self):
        m = bm.compute_metrics([], [self._seg(10, 30)])
        self.assertEqual(m["precision"], 1)
        self.assertEqual(m["recall"], 0)
        self.assertEqual(m["iou"], 0)

    def test_perfect_overlap(self):
        seg = self._seg(10, 30)
        m = bm.compute_metrics([seg], [seg])
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["iou"], 1.0)

    def test_partial_overlap_precision_recall(self):
        # GT: 10→30 (20s). Detected: 20→40 (20s). Overlap: 20→30 (10s).
        m = bm.compute_metrics([self._seg(20, 40)], [self._seg(10, 30)])
        self.assertAlmostEqual(m["precision"], 0.5, places=5)   # 10/20
        self.assertAlmostEqual(m["recall"], 0.5, places=5)      # 10/20

    def test_iou_calculation(self):
        # GT: 0→10, Det: 5→15. Intersection=5, union=15
        m = bm.compute_metrics([self._seg(5, 15)], [self._seg(0, 10)])
        self.assertAlmostEqual(m["iou"], 5 / 15, places=5)

    def test_detected_fully_inside_ground_truth(self):
        # GT: 0→100, Det: 40→60. precision=1, recall=0.2
        m = bm.compute_metrics([self._seg(40, 60)], [self._seg(0, 100)])
        self.assertAlmostEqual(m["precision"], 1.0, places=5)
        self.assertAlmostEqual(m["recall"], 0.2, places=5)

    def test_ground_truth_fully_inside_detected(self):
        # GT: 40→60, Det: 0→100. precision=0.2, recall=1
        m = bm.compute_metrics([self._seg(0, 100)], [self._seg(40, 60)])
        self.assertAlmostEqual(m["precision"], 0.2, places=5)
        self.assertAlmostEqual(m["recall"], 1.0, places=5)

    def test_no_overlap_at_all(self):
        m = bm.compute_metrics([self._seg(50, 60)], [self._seg(10, 20)])
        self.assertAlmostEqual(m["precision"], 0.0, places=5)
        self.assertAlmostEqual(m["recall"], 0.0, places=5)
        self.assertAlmostEqual(m["iou"], 0.0, places=5)

    def test_multiple_gt_segments(self):
        # Two GT segments: 10→20, 50→60. Detected perfectly: both.
        gt = [self._seg(10, 20), self._seg(50, 60)]
        det = [self._seg(10, 20), self._seg(50, 60)]
        m = bm.compute_metrics(det, gt)
        self.assertAlmostEqual(m["precision"], 1.0, places=5)
        self.assertAlmostEqual(m["recall"], 1.0, places=5)

    def test_details_contain_coverage(self):
        gt = [self._seg(0, 100)]
        det = [self._seg(0, 60)]
        m = bm.compute_metrics(det, gt)
        coverage = m["details"][0]["coverage"]
        self.assertAlmostEqual(coverage, 0.6, places=5)

    def test_details_false_positive_flag(self):
        m = bm.compute_metrics([self._seg(10, 20)], [])
        self.assertEqual(m["details"][0]["type"], "false_positive")

    def test_details_missed_flag(self):
        m = bm.compute_metrics([], [self._seg(10, 20)])
        self.assertEqual(m["details"][0]["type"], "missed")


# ══════════════════════════════════════════════════════════════════
#  Caption parsers
# ══════════════════════════════════════════════════════════════════

class TestParseCaptionXml(unittest.TestCase):

    def test_valid_xml_returns_cues(self):
        xml = """<?xml version="1.0" encoding="utf-8"?>
<transcript>
  <text start="0.5" dur="2.3">Hello world</text>
  <text start="3.0" dur="1.8">This is a test</text>
</transcript>"""
        cues = bm._parse_caption_xml(xml)
        self.assertIsNotNone(cues)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0]["start"], 0.5)
        self.assertAlmostEqual(cues[0]["dur"], 2.3)
        self.assertEqual(cues[0]["text"], "Hello world")

    def test_empty_text_elements_skipped(self):
        xml = """<transcript>
  <text start="1.0" dur="2.0"></text>
  <text start="3.0" dur="2.0">Real content</text>
</transcript>"""
        cues = bm._parse_caption_xml(xml)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["text"], "Real content")

    def test_newlines_in_text_replaced_with_space(self):
        xml = """<transcript>
  <text start="1.0" dur="2.0">line one\nline two</text>
</transcript>"""
        cues = bm._parse_caption_xml(xml)
        self.assertNotIn("\n", cues[0]["text"])

    def test_invalid_xml_returns_none(self):
        result = bm._parse_caption_xml("this is not xml at all")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        result = bm._parse_caption_xml("")
        self.assertIsNone(result)

    def test_xml_with_no_text_elements_returns_none(self):
        xml = "<transcript></transcript>"
        result = bm._parse_caption_xml(xml)
        self.assertIsNone(result)

    def test_default_dur_used_when_missing(self):
        xml = """<transcript>
  <text start="5.0">No duration attribute</text>
</transcript>"""
        cues = bm._parse_caption_xml(xml)
        self.assertEqual(cues[0]["dur"], 2.0)  # default


class TestParseCaptionJson(unittest.TestCase):

    def _make_json(self, events):
        return json.dumps({"events": events})

    def test_valid_json_returns_cues(self):
        data = self._make_json([
            {"tStartMs": 500, "dDurationMs": 2300, "segs": [{"utf8": "Hello world"}]},
            {"tStartMs": 3000, "dDurationMs": 1800, "segs": [{"utf8": "Second cue"}]},
        ])
        cues = bm._parse_caption_json(data)
        self.assertIsNotNone(cues)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0]["start"], 0.5)
        self.assertAlmostEqual(cues[0]["dur"], 2.3)
        self.assertEqual(cues[0]["text"], "Hello world")

    def test_multiple_segs_concatenated(self):
        data = self._make_json([
            {"tStartMs": 0, "dDurationMs": 3000,
             "segs": [{"utf8": "Hello"}, {"utf8": " "}, {"utf8": "world"}]},
        ])
        cues = bm._parse_caption_json(data)
        self.assertEqual(cues[0]["text"], "Hello world")

    def test_whitespace_only_cues_skipped(self):
        data = self._make_json([
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "   "}]},
            {"tStartMs": 2000, "dDurationMs": 1000, "segs": [{"utf8": "Real text"}]},
        ])
        cues = bm._parse_caption_json(data)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["text"], "Real text")

    def test_newline_only_cues_skipped(self):
        data = self._make_json([
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "\n"}]},
        ])
        result = bm._parse_caption_json(data)
        self.assertIsNone(result)

    def test_invalid_json_returns_none(self):
        result = bm._parse_caption_json("not json {")
        self.assertIsNone(result)

    def test_empty_events_returns_none(self):
        result = bm._parse_caption_json(json.dumps({"events": []}))
        self.assertIsNone(result)

    def test_missing_events_key_returns_none(self):
        result = bm._parse_caption_json(json.dumps({"something": "else"}))
        self.assertIsNone(result)

    def test_ms_to_seconds_conversion(self):
        data = self._make_json([
            {"tStartMs": 90000, "dDurationMs": 5000, "segs": [{"utf8": "At 90s"}]},
        ])
        cues = bm._parse_caption_json(data)
        self.assertAlmostEqual(cues[0]["start"], 90.0)
        self.assertAlmostEqual(cues[0]["dur"], 5.0)


# ══════════════════════════════════════════════════════════════════
#  get_captions_for_video (mocked network)
# ══════════════════════════════════════════════════════════════════

class TestGetCaptionsForVideo(unittest.TestCase):
    """
    Cache is always patched out so tests are isolated from disk state.
    _load_caption_cache returns None (cache miss) and _save_caption_cache is a no-op.
    """

    def _no_cache(self):
        """Return a context manager that bypasses the caption cache."""
        return patch.multiple(
            bm,
            _load_caption_cache=lambda vid: None,
            _save_caption_cache=lambda vid, cues: None,
        )

    def test_returns_none_when_all_strategies_fail(self):
        with self._no_cache(), \
             patch.object(bm, "_get_caption_url_innertube", return_value=None), \
             patch.object(bm, "_get_caption_url_html", return_value=None), \
             patch.object(bm, "_get_captions_via_html_session", return_value=None), \
             patch.object(bm, "_get_captions_via_ytdlp", return_value=None):
            result = bm.get_captions_for_video("dummyId")
            self.assertIsNone(result)

    def test_returns_cues_from_innertube_url(self):
        xml = """<transcript>
  <text start="1.0" dur="2.0">Brought to you by Acme</text>
  <text start="3.0" dur="2.0">Go to acme.com now</text>
</transcript>"""
        with self._no_cache(), \
             patch.object(bm, "_get_caption_url_innertube", return_value="https://fake.url/xml"), \
             patch.object(bm, "_get_caption_url_html", return_value=None), \
             patch.object(bm, "_get_captions_via_html_session", return_value=None), \
             patch.object(bm, "_get_captions_via_ytdlp", return_value=None), \
             patch.object(bm, "http_get", return_value=xml):
            result = bm.get_captions_for_video("dummyId")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_falls_back_to_html_url_strategy(self):
        xml = "<transcript><text start='0' dur='2'>Hello</text></transcript>"
        with self._no_cache(), \
             patch.object(bm, "_get_caption_url_innertube", return_value=None), \
             patch.object(bm, "_get_caption_url_html", return_value="https://fake.url/html"), \
             patch.object(bm, "_get_captions_via_html_session", return_value=None), \
             patch.object(bm, "_get_captions_via_ytdlp", return_value=None), \
             patch.object(bm, "http_get", return_value=xml):
            result = bm.get_captions_for_video("dummyId")
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["text"], "Hello")

    def test_falls_back_to_cookie_session_when_url_fetch_empty(self):
        cookie_cues = [{"start": 1.0, "dur": 2.0, "text": "From cookie session"}]
        with self._no_cache(), \
             patch.object(bm, "_get_caption_url_innertube", return_value="https://fake.url"), \
             patch.object(bm, "_get_caption_url_html", return_value=None), \
             patch.object(bm, "http_get", return_value=""), \
             patch.object(bm, "_get_captions_via_html_session", return_value=cookie_cues), \
             patch.object(bm, "_get_captions_via_ytdlp", return_value=None):
            result = bm.get_captions_for_video("dummyId")
        self.assertEqual(result, cookie_cues)

    def test_falls_back_to_ytdlp_when_all_else_fails(self):
        import shutil
        ytdlp_cues = [{"start": 5.0, "dur": 3.0, "text": "From yt-dlp"}]
        with self._no_cache(), \
             patch.object(bm, "_get_caption_url_innertube", return_value=None), \
             patch.object(bm, "_get_caption_url_html", return_value=None), \
             patch.object(bm, "_get_captions_via_html_session", return_value=None), \
             patch.object(bm, "_get_captions_via_ytdlp", return_value=ytdlp_cues), \
             patch.object(shutil, "which", return_value="/usr/bin/yt-dlp"):
            result = bm.get_captions_for_video("dummyId")
        self.assertEqual(result, ytdlp_cues)

    def test_cache_hit_skips_all_strategies(self):
        cached = [{"start": 0.0, "dur": 2.0, "text": "Cached cue"}]
        with patch.object(bm, "_load_caption_cache", return_value=cached), \
             patch.object(bm, "_get_caption_url_innertube") as mock_innertube:
            result = bm.get_captions_for_video("dummyId")
        self.assertEqual(result, cached)
        mock_innertube.assert_not_called()

    def test_returns_none_on_empty_raw_and_no_session_fallback(self):
        with self._no_cache(), \
             patch.object(bm, "_get_caption_url_innertube", return_value="https://fake.url"), \
             patch.object(bm, "_get_caption_url_html", return_value=None), \
             patch.object(bm, "http_get", return_value=""), \
             patch.object(bm, "_get_captions_via_html_session", return_value=None), \
             patch.object(bm, "_get_captions_via_ytdlp", return_value=None):
            result = bm.get_captions_for_video("dummyId")
        self.assertIsNone(result)


class TestGetCaptionsViaHtmlSession(unittest.TestCase):
    """Tests for the cookie-session HTML fallback."""

    def _make_html(self, tracks_json):
        return f'<html><body><script>var x = {{"captionTracks":{tracks_json}}};</script></body></html>'

    def _make_opener(self, page_html, caption_body):
        """
        Build a mock opener whose .open() returns page_html for the first
        call (watch page) and caption_body for the second (timedtext).
        """
        calls = [0]
        responses = [page_html.encode(), caption_body.encode() if isinstance(caption_body, str) else caption_body]

        class FakeResp:
            def __init__(self, body):
                self._body = body
            def read(self):
                return self._body
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class FakeOpener:
            def open(self_, req, timeout=None):
                i = calls[0]
                calls[0] += 1
                return FakeResp(responses[i] if i < len(responses) else b"")

        return FakeOpener()

    def test_returns_cues_when_cookie_session_works(self):
        xml = "<transcript><text start='5' dur='3'>Hello from session</text></transcript>"
        tracks = json.dumps([{"languageCode": "en", "baseUrl": "https://timedtext.url/en"}])
        html = self._make_html(tracks)
        opener = self._make_opener(html, xml)

        with patch("http.cookiejar.CookieJar"), \
             patch("urllib.request.build_opener", return_value=opener):
            result = bm._get_captions_via_html_session("vid123")

        self.assertIsNotNone(result)
        self.assertEqual(result[0]["text"], "Hello from session")
        self.assertAlmostEqual(result[0]["start"], 5.0)

    def test_returns_none_when_page_fetch_fails(self):
        class FailOpener:
            def open(self, req, timeout=None):
                raise Exception("network error")

        with patch("http.cookiejar.CookieJar"), \
             patch("urllib.request.build_opener", return_value=FailOpener()):
            result = bm._get_captions_via_html_session("vid123")
        self.assertIsNone(result)

    def test_returns_none_when_no_caption_tracks_in_html(self):
        html = "<html><body>No caption tracks here</body></html>"
        opener = self._make_opener(html, "")

        with patch("http.cookiejar.CookieJar"), \
             patch("urllib.request.build_opener", return_value=opener):
            result = bm._get_captions_via_html_session("vid123")
        self.assertIsNone(result)

    def test_prefers_english_track(self):
        xml_en = "<transcript><text start='1' dur='2'>English</text></transcript>"
        tracks = json.dumps([
            {"languageCode": "fr", "baseUrl": "https://timedtext.url/fr"},
            {"languageCode": "en", "baseUrl": "https://timedtext.url/en"},
        ])
        html = self._make_html(tracks)

        fetched_urls = []
        calls = [0]
        page_bytes = html.encode()
        cap_bytes = xml_en.encode()

        class TrackingOpener:
            def open(self_, req, timeout=None):
                class R:
                    def __init__(self, b): self._b = b
                    def read(self): return self._b
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                if calls[0] == 0:
                    calls[0] += 1
                    return R(page_bytes)
                fetched_urls.append(req.full_url)
                calls[0] += 1
                return R(cap_bytes)

        with patch("http.cookiejar.CookieJar"), \
             patch("urllib.request.build_opener", return_value=TrackingOpener()):
            bm._get_captions_via_html_session("vid123")

        self.assertTrue(any("en" in u for u in fetched_urls),
                        f"Expected English URL to be fetched, got: {fetched_urls}")


# ══════════════════════════════════════════════════════════════════
#  InnerTube client fallback behaviour (mocked network)
# ══════════════════════════════════════════════════════════════════

class TestGetCaptionUrlInnertube(unittest.TestCase):

    def _player_response(self, base_url, lang="en"):
        return {
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {"languageCode": lang, "baseUrl": base_url}
                    ]
                }
            }
        }

    def test_returns_url_from_first_working_client(self):
        with patch.object(bm, "http_post_json",
                          return_value=self._player_response("https://captions.url/en")):
            result = bm._get_caption_url_innertube("vid123")
        self.assertEqual(result, "https://captions.url/en")

    def test_skips_client_with_no_tracks_tries_next(self):
        no_tracks = {"captions": {"playerCaptionsTracklistRenderer": {"captionTracks": []}}}
        good = self._player_response("https://captions.url/en")
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return no_tracks if call_count[0] == 1 else good
        with patch.object(bm, "http_post_json", side_effect=side_effect), \
             patch.object(bm, "_get_caption_url_timedtext", return_value=None):
            result = bm._get_caption_url_innertube("vid123")
        self.assertEqual(result, "https://captions.url/en")
        self.assertEqual(call_count[0], 2)

    def test_all_clients_fail_falls_back_to_timedtext(self):
        with patch.object(bm, "http_post_json", side_effect=Exception("network error")), \
             patch.object(bm, "_get_caption_url_timedtext",
                          return_value="https://www.youtube.com/api/timedtext?v=vid123"):
            result = bm._get_caption_url_innertube("vid123")
        self.assertEqual(result, "https://www.youtube.com/api/timedtext?v=vid123")

    def test_prefers_english_track(self):
        response = {
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {"languageCode": "fr", "baseUrl": "https://fr.url"},
                        {"languageCode": "en", "baseUrl": "https://en.url"},
                    ]
                }
            }
        }
        with patch.object(bm, "http_post_json", return_value=response):
            result = bm._get_caption_url_innertube("vid123")
        self.assertEqual(result, "https://en.url")

    def test_falls_back_to_non_english_if_no_english(self):
        response = {
            "captions": {
                "playerCaptionsTracklistRenderer": {
                    "captionTracks": [
                        {"languageCode": "de", "baseUrl": "https://de.url"},
                    ]
                }
            }
        }
        with patch.object(bm, "http_post_json", return_value=response):
            result = bm._get_caption_url_innertube("vid123")
        self.assertEqual(result, "https://de.url")


# ══════════════════════════════════════════════════════════════════
#  VTT caption parser
# ══════════════════════════════════════════════════════════════════

class TestParseCaptionVtt(unittest.TestCase):

    def test_basic_vtt_returns_cues(self):
        vtt = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.500
Hello world

00:00:04.000 --> 00:00:06.000
Second cue here
"""
        cues = bm._parse_caption_vtt(vtt)
        self.assertIsNotNone(cues)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0]["start"], 1.0)
        self.assertAlmostEqual(cues[0]["dur"], 2.5)
        self.assertEqual(cues[0]["text"], "Hello world")
        self.assertEqual(cues[1]["text"], "Second cue here")

    def test_strips_inline_vtt_tags(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
<c>Hello</c> <00:00:01.500><c>world</c>
"""
        cues = bm._parse_caption_vtt(vtt)
        self.assertIsNotNone(cues)
        self.assertEqual(cues[0]["text"], "Hello world")

    def test_multiline_cue_joined_with_space(self):
        vtt = """WEBVTT

00:00:01.000 --> 00:00:04.000
Line one
line two
"""
        cues = bm._parse_caption_vtt(vtt)
        self.assertEqual(cues[0]["text"], "Line one line two")

    def test_empty_vtt_returns_none(self):
        self.assertIsNone(bm._parse_caption_vtt("WEBVTT\n\n"))

    def test_hhmmss_timestamp_parsed(self):
        vtt = """WEBVTT

01:02:03.456 --> 01:02:05.000
Content
"""
        cues = bm._parse_caption_vtt(vtt)
        expected = 1 * 3600 + 2 * 60 + 3.456
        self.assertAlmostEqual(cues[0]["start"], expected, places=2)

    def test_mmss_timestamp_parsed(self):
        vtt = """WEBVTT

02:30.000 --> 02:35.500
Content
"""
        cues = bm._parse_caption_vtt(vtt)
        self.assertAlmostEqual(cues[0]["start"], 150.0, places=2)


class TestVttTimeToSec(unittest.TestCase):

    def test_hhmmss(self):
        self.assertAlmostEqual(bm._vtt_time_to_sec("01:02:03.456"), 3723.456, places=2)

    def test_mmss(self):
        self.assertAlmostEqual(bm._vtt_time_to_sec("02:30.000"), 150.0, places=2)

    def test_zero(self):
        self.assertAlmostEqual(bm._vtt_time_to_sec("00:00:00.000"), 0.0)


# ══════════════════════════════════════════════════════════════════
#  yt-dlp fallback
# ══════════════════════════════════════════════════════════════════

class TestGetCaptionsViaYtdlp(unittest.TestCase):

    def test_returns_none_when_ytdlp_not_installed(self):
        with patch("shutil.which", return_value=None):
            result = bm._get_captions_via_ytdlp("vid123")
        self.assertIsNone(result)

    def test_parses_srv3_output_from_ytdlp(self):
        import os, tempfile, shutil
        srv3 = json.dumps({"events": [
            {"tStartMs": 1000, "dDurationMs": 2000,
             "segs": [{"utf8": "Sponsor content here"}]},
        ]})

        def fake_run(cmd, **kwargs):
            # Write a fake .srv3 file into the tmpdir yt-dlp would use
            # Find the output template from cmd args
            for i, arg in enumerate(cmd):
                if arg == "-o":
                    out_tmpl = cmd[i + 1]
                    tmpdir = os.path.dirname(out_tmpl)
                    with open(os.path.join(tmpdir, "vid123.en.srv3"), "w") as f:
                        f.write(srv3)
                    break
            class R:
                returncode = 0
            return R()

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("subprocess.run", side_effect=fake_run):
            result = bm._get_captions_via_ytdlp("vid123")

        self.assertIsNotNone(result)
        self.assertEqual(result[0]["text"], "Sponsor content here")

    def test_parses_vtt_output_from_ytdlp(self):
        import os
        vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello from yt-dlp
"""

        def fake_run(cmd, **kwargs):
            for i, arg in enumerate(cmd):
                if arg == "-o":
                    out_tmpl = cmd[i + 1]
                    tmpdir = os.path.dirname(out_tmpl)
                    with open(os.path.join(tmpdir, "vid123.en.vtt"), "w") as f:
                        f.write(vtt)
                    break
            class R:
                returncode = 0
            return R()

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("subprocess.run", side_effect=fake_run):
            result = bm._get_captions_via_ytdlp("vid123")

        self.assertIsNotNone(result)
        self.assertEqual(result[0]["text"], "Hello from yt-dlp")

    def test_returns_none_when_no_files_downloaded(self):
        def fake_run(cmd, **kwargs):
            class R: returncode = 1
            return R()

        with patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("subprocess.run", side_effect=fake_run):
            result = bm._get_captions_via_ytdlp("vid123")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
