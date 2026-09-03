/**
 * Copy text in both secure browsers and plain-HTTP LAN deployments.
 *
 * The asynchronous Clipboard API is normally restricted to HTTPS/localhost.
 * DSA is also commonly served from an HTTP reverse proxy on a local network,
 * so fall back to the legacy selection + execCommand path when needed.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof text !== 'string') {
    return false;
  }

  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Continue to the plain-HTTP fallback below.
    }
  }

  if (typeof document === 'undefined' || typeof document.execCommand !== 'function') {
    return false;
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.setAttribute('aria-hidden', 'true');
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  textarea.style.top = '0';
  textarea.style.opacity = '0';

  const activeElement = document.activeElement instanceof HTMLElement
    ? document.activeElement
    : null;
  const selection = typeof window !== 'undefined' ? window.getSelection() : null;
  const savedRanges: Range[] = [];
  if (selection) {
    for (let index = 0; index < selection.rangeCount; index += 1) {
      savedRanges.push(selection.getRangeAt(index).cloneRange());
    }
  }

  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);

  let copied = false;
  try {
    copied = document.execCommand('copy');
  } catch {
    copied = false;
  } finally {
    textarea.remove();
    if (selection) {
      selection.removeAllRanges();
      savedRanges.forEach((range) => selection.addRange(range));
    }
    activeElement?.focus();
  }

  return copied;
}
