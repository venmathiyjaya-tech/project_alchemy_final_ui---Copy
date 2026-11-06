/* ═══════════════════════════════════════════════════════════════════════════
   VASYA — Project Alchemy  ·  main.js
   Global JavaScript — loaded on every page via base.html
═══════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  /* ── 1. Global namespace ──────────────────────────────────────────── */
  window.VASYA = window.VASYA || {};

  /* ── 2. API helpers ───────────────────────────────────────────────── */
  VASYA.apiPost = async function (url, body) {
    const res  = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  };

  VASYA.apiDelete = async function (url) {
    const res  = await fetch(url, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Delete failed (${res.status})`);
    return data;
  };

  /* ── 3. Notification toast ────────────────────────────────────────── */
  VASYA.showNotification = function (message, type) {
    type = type || 'info';
    let stack = document.getElementById('vasya-notif-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'vasya-notif-stack';
      Object.assign(stack.style, {
        position:      'fixed',
        bottom:        '2rem',
        left:          '50%',
        transform:     'translateX(-50%)',
        display:       'flex',
        flexDirection: 'column',
        gap:           '0.5rem',
        zIndex:        '9999',
        width:         'min(460px, 90vw)',
        pointerEvents: 'none',
      });
      document.body.appendChild(stack);
    }

    const colors = {
      success: { border: '#4a8a60', bg: 'rgba(28,48,33,0.96)', text: '#8cc4a0' },
      error:   { border: '#7a3030', bg: 'rgba(48,12,12,0.97)', text: '#cc8888' },
      info:    { border: '#c8a84a', bg: 'rgba(28,20,8,0.97)',  text: '#d0c0a0' },
    };
    const c = colors[type] || colors.info;

    // Inject keyframes once
    if (!document.getElementById('vasya-notif-kf')) {
      const s = document.createElement('style');
      s.id = 'vasya-notif-kf';
      s.textContent = `
        @keyframes nIn  { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
        @keyframes nOut { from{opacity:1} to{opacity:0;transform:translateY(6px)} }
      `;
      document.head.appendChild(s);
    }

    const el = document.createElement('div');
    Object.assign(el.style, {
      fontFamily:     "'EB Garamond', Georgia, serif",
      fontStyle:      'italic',
      fontSize:       '0.9rem',
      lineHeight:     '1.6',
      color:          c.text,
      background:     c.bg,
      border:         `1px solid ${c.border}`,
      borderLeft:     `2px solid ${c.border}`,
      padding:        '0.85rem 1.25rem',
      pointerEvents:  'auto',
      animation:      'nIn 320ms cubic-bezier(0.16,1,0.3,1) both',
      backdropFilter: 'blur(10px)',
    });
    el.textContent = message;
    stack.appendChild(el);

    setTimeout(() => {
      el.style.animation = 'nOut 300ms ease forwards';
      setTimeout(() => el.remove(), 320);
    }, 5000);
  };

  /* ── 4. Flash dismiss ─────────────────────────────────────────────── */
  document.addEventListener('click', e => {
    const btn = e.target.closest('[data-action="dismiss-flash"]');
    if (!btn) return;
    const flash = btn.closest('.flash');
    if (flash) {
      flash.style.transition = 'opacity 280ms';
      flash.style.opacity = '0';
      setTimeout(() => flash.remove(), 300);
    }
  });

  /* ── 5. Mobile nav burger ─────────────────────────────────────────── */
  const burger   = document.querySelector('.nav-burger');
  const navLinks = document.getElementById('nav-links');

  if (burger && navLinks) {
    burger.addEventListener('click', () => {
      const open = burger.getAttribute('aria-expanded') === 'true';
      burger.setAttribute('aria-expanded', String(!open));
      navLinks.classList.toggle('is-open', !open);
    });
    document.addEventListener('click', e => {
      if (!burger.contains(e.target) && !navLinks.contains(e.target)) {
        burger.setAttribute('aria-expanded', 'false');
        navLinks.classList.remove('is-open');
      }
    });
    navLinks.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        burger.setAttribute('aria-expanded', 'false');
        navLinks.classList.remove('is-open');
      });
    });
  }

  /* ── 6. Scroll-based nav border ───────────────────────────────────── */
  const navHeader = document.querySelector('.nav-header');
  if (navHeader) {
    const updateNav = () => {
      navHeader.style.borderBottomColor = window.scrollY > 20
        ? 'rgba(200,168,74,0.22)'
        : 'rgba(200,168,74,0.08)';
    };
    window.addEventListener('scroll', updateNav, { passive: true });
    updateNav();
  }

  /* ── 7. Intersection observer — fade-in panels ────────────────────── */
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('in-view');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.08 });

    // Inject in-view style once
    if (!document.getElementById('vasya-inview-style')) {
      const s = document.createElement('style');
      s.id = 'vasya-inview-style';
      s.textContent = '.in-view{opacity:1!important;transform:none!important;}';
      document.head.appendChild(s);
    }

    document.querySelectorAll(
      '.obs-panel, .scent-card, .entry-item, .landing-pillar, .archive-card'
    ).forEach(el => {
      el.style.opacity    = '0';
      el.style.transform  = 'translateY(18px)';
      el.style.transition = 'opacity 520ms cubic-bezier(0.16,1,0.3,1), transform 520ms cubic-bezier(0.16,1,0.3,1)';
      io.observe(el);
    });
  }

  /* ── 8. VAD bar animated fill ─────────────────────────────────────── */
  document.querySelectorAll('.vad-bar-fill').forEach(fill => {
    const target = fill.style.width;
    fill.style.width = '0%';
    requestAnimationFrame(() => {
      setTimeout(() => { fill.style.width = target; }, 250);
    });
  });

  /* ── 9. Textarea auto-resize ──────────────────────────────────────── */
  document.querySelectorAll('.journal-textarea').forEach(ta => {
    function resize() {
      ta.style.height = 'auto';
      ta.style.height = Math.max(ta.scrollHeight, 240) + 'px';
    }
    ta.addEventListener('input', resize);
    resize();
  });

  /* ── 10. Journal surface ambient glow ────────────────────────────── */
  const surface  = document.getElementById('compose-surface');
  const textarea = document.getElementById('journal-textarea');
  if (surface && textarea) {
    textarea.addEventListener('focus', () => {
      surface.style.boxShadow = '0 0 80px rgba(200,168,74,0.06), 0 8px 32px rgba(0,0,0,0.45)';
    });
    textarea.addEventListener('blur', () => {
      surface.style.boxShadow = '';
    });
  }

  /* ── 11. Smooth page transitions ─────────────────────────────────── */
  // Skip fade transitions on auth pages — the body fade hides flash messages
  // before the user can read them.
  const isAuthPage = document.getElementById('auth-unified') !== null;

  if (!isAuthPage) {
    if (!document.getElementById('vasya-transition-style')) {
      const s = document.createElement('style');
      s.id = 'vasya-transition-style';
      s.textContent = `
        body.page-leaving {
          opacity: 0;
          transition: opacity 300ms ease;
        }
      `;
      document.head.appendChild(s);
    }

    document.addEventListener('click', e => {
      const link = e.target.closest('a[href]');
      if (!link) return;
      const href = link.getAttribute('href');
      if (!href || href.startsWith('#') || href.startsWith('http') ||
          href.startsWith('mailto') || link.hasAttribute('download') ||
          link.target === '_blank') return;
      e.preventDefault();
      document.body.classList.add('page-leaving');
      setTimeout(() => { window.location.href = href; }, 260);
    });

    // Fade in on page load (non-auth pages only)
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 380ms ease';
    requestAnimationFrame(() => {
      setTimeout(() => { document.body.style.opacity = '1'; }, 30);
    });
  }

})();