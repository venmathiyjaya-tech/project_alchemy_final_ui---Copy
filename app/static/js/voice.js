/* ═══════════════════════════════════════════════════════════════════════════
   VASYA — Project Alchemy  ·  voice.js
   Voice journaling (speech → text) and read-aloud (text → speech) built on the
   browser's Web Speech API. No server code and no audio storage: the transcript
   simply lands in the existing journal textarea, so everything downstream
   (emotion model, VAD, recommender) is unchanged.

   Browser notes
   - Needs a secure context: https:// or http://localhost (not a LAN IP over http).
   - Chrome / Edge send the audio to the browser vendor's speech service.
   - Firefox has no speech recognition; the UI degrades gracefully.
═══════════════════════════════════════════════════════════════════════════ */
(function (root) {
  'use strict';

  const VOICE = { active: null };
  const SENTENCE_END = /[.!?…]["')\]]?\s*$/;

  /* ── pure helpers (unit-tested in Node) ─────────────────────────────── */

  VOICE.normalise = function (s) {
    return String(s || '').replace(/\s+/g, ' ').trim();
  };

  /**
   * Combine what was already in the box (`base`) with the spoken words.
   * Returns { text, truncated }; never longer than maxLen, never cuts mid-word.
   */
  VOICE.mergeTranscript = function (base, finalText, interimText, maxLen) {
    base = String(base || '');
    let spoken = VOICE.normalise(VOICE.normalise(finalText) + ' ' + VOICE.normalise(interimText));
    if (!spoken) return { text: base, truncated: false };

    const baseTrim = base.replace(/\s+$/, '');
    if (!baseTrim || SENTENCE_END.test(baseTrim) || /\n\s*$/.test(base)) {
      spoken = spoken.charAt(0).toUpperCase() + spoken.slice(1);
    }

    let prefix;
    if (!baseTrim)                     prefix = '';
    else if (/\n\s*$/.test(base))      prefix = base.replace(/[ \t]+$/, '');
    else                               prefix = baseTrim + ' ';

    let text = prefix + spoken;
    let truncated = false;
    if (maxLen && text.length > maxLen) {
      truncated = true;
      let cut = text.slice(0, maxLen);
      const lastSpace = cut.lastIndexOf(' ');
      if (lastSpace > prefix.length) cut = cut.slice(0, lastSpace);
      text = cut.replace(/\s+$/, '');
    }
    return { text, truncated };
  };

  VOICE.isSupported = function (r) {
    r = r || root;
    return {
      stt: !!(r.SpeechRecognition || r.webkitSpeechRecognition),
      tts: !!(r.speechSynthesis && r.SpeechSynthesisUtterance),
    };
  };

  /* ── speech → text ──────────────────────────────────────────────────── */

  const ERROR_MESSAGES = {
    'not-allowed':            'Microphone access was blocked. Allow it from the address bar, then try again.',
    'service-not-allowed':    'Microphone access was blocked. Allow it from the address bar, then try again.',
    'audio-capture':          'No microphone was found.',
    'network':                "The browser's speech service could not be reached.",
    'language-not-supported': 'That language is not supported for voice input.',
  };

  /**
   * opts: { textarea, lang, maxLen, onState(state, detail), SpeechRecognition? }
   * state is 'listening' | 'idle' | 'error'. Returns null when unsupported.
   */
  VOICE.createDictation = function (opts) {
    const SR = opts.SpeechRecognition || root.SpeechRecognition || root.webkitSpeechRecognition;
    if (!SR) return null;

    const ta     = opts.textarea;
    const maxLen = opts.maxLen || 2000;
    let lang = opts.lang || 'en-US';

    let rec = null, listening = false, acceptLate = false;
    let base = '', finalText = '', interim = '';
    let restartTimer = null, quickEnds = 0, startedAt = 0, selfUpdate = false;

    const emit = (state, detail) => { if (opts.onState) opts.onState(state, detail || ''); };

    function render() {
      const merged = VOICE.mergeTranscript(base, finalText, interim, maxLen);
      selfUpdate = true;
      ta.value = merged.text;
      if (typeof ta.dispatchEvent === 'function' && typeof root.Event === 'function') {
        ta.dispatchEvent(new root.Event('input', { bubbles: true }));
      }
      selfUpdate = false;
      return merged;
    }

    // If the user types or edits while dictating, treat their text as the new base.
    if (typeof ta.addEventListener === 'function') {
      ta.addEventListener('input', () => {
        if (selfUpdate || !listening) return;
        base = ta.value; finalText = ''; interim = '';
      });
    }

    function end(reason, message) {           // error / limit: stop immediately
      if (!listening) return;
      listening = false; acceptLate = false;
      clearTimeout(restartTimer);
      interim = '';
      try { rec && rec.abort(); } catch (_) { /* ignore */ }
      render();
      emit(reason === 'error' ? 'error' : 'idle',
           message || (reason === 'limit' ? 'Character limit reached.' : ''));
    }

    function build() {
      const r = new SR();
      r.lang = lang;
      r.continuous = true;
      r.interimResults = true;
      r.maxAlternatives = 1;

      r.onstart = () => { startedAt = Date.now(); };

      r.onresult = (e) => {
        if (!listening && !acceptLate) return;
        quickEnds = 0;
        interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const res = e.results[i];
          const t = res && res[0] ? res[0].transcript : '';
          if (res.isFinal) finalText += ' ' + t; else interim += t;
        }
        const merged = render();
        if (merged.truncated) end('limit');
      };

      r.onerror = (e) => {
        const code = e && e.error;
        if (code === 'no-speech' || code === 'aborted') return;   // normal; onend restarts
        end('error', ERROR_MESSAGES[code] || ('Voice input stopped (' + code + ').'));
      };

      r.onend = () => {
        if (!listening) return;
        // Chrome ends a session after a pause; keep going until the user presses Stop.
        quickEnds = (Date.now() - startedAt) < 1500 ? quickEnds + 1 : 0;
        if (quickEnds >= 4) { end('error', 'Voice input keeps stopping. Please try again.'); return; }
        restartTimer = setTimeout(() => {
          if (listening) { try { r.start(); } catch (_) { /* already started */ } }
        }, 250);
      };
      return r;
    }

    const ctrl = {
      start() {
        if (listening) return;
        base = ta.value; finalText = ''; interim = ''; quickEnds = 0;
        listening = true; acceptLate = false;
        rec = build();
        try { rec.start(); }
        catch (_) { listening = false; emit('error', 'Could not start voice input.'); return; }
        emit('listening');
      },

      /** Graceful stop keeps the last phrase; {discard:true} drops anything pending. */
      stop(o) {
        if (!listening) return;
        listening = false;
        clearTimeout(restartTimer);
        if (o && o.discard) {
          acceptLate = false;
          try { rec.abort(); } catch (_) { /* ignore */ }
          interim = '';
        } else {
          acceptLate = true;
          setTimeout(() => { acceptLate = false; }, 1500);
          try { rec.stop(); } catch (_) { /* ignore */ }
          interim = '';
          render();
        }
        emit('idle');
      },

      toggle() { if (listening) ctrl.stop(); else ctrl.start(); },
      isListening() { return listening; },
      setLang(l) { lang = l; if (rec) rec.lang = l; },
    };

    VOICE.active = ctrl;
    return ctrl;
  };

  /* ── text → speech ──────────────────────────────────────────────────── */

  VOICE.speak = function (text, lang) {
    const synth = root.speechSynthesis;
    if (!synth || typeof root.SpeechSynthesisUtterance !== 'function') return false;
    const clean = VOICE.normalise(text);
    if (!clean) return false;
    synth.cancel();
    const u = new root.SpeechSynthesisUtterance(clean);
    u.lang = lang || 'en-US';
    u.rate = 0.95;
    synth.speak(u);
    return true;
  };

  VOICE.stopSpeaking = function () {
    if (root.speechSynthesis) root.speechSynthesis.cancel();
  };

  /** Stop dictation (discarding anything pending) and any speech. */
  VOICE.stopAll = function () {
    if (VOICE.active) VOICE.active.stop({ discard: true });
    VOICE.stopSpeaking();
  };

  /* ── export ─────────────────────────────────────────────────────────── */
  root.VASYA = root.VASYA || {};
  root.VASYA.voice = VOICE;
  if (typeof module !== 'undefined' && module.exports) module.exports = VOICE;

})(typeof window !== 'undefined' ? window : globalThis);
