import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Copy, ExternalLink, KeyRound, RefreshCw } from 'lucide-react';
import { systemConfigApi } from '../../api/systemConfig';
import { getParsedApiError, type ParsedApiError } from '../../api/error';
import { useUiLanguage } from '../../contexts/UiLanguageContext';
import { copyToClipboard } from '../../utils/clipboard';
import type { CodexOAuthLoginStart, CodexOAuthStatus } from '../../types/systemConfig';
import { CODEX_OAUTH_PANEL_DOM_ID } from './llmProviderTemplates';
import { ApiErrorAlert, Badge, Button } from '../common';
import { SettingsAlert } from './SettingsAlert';

interface CodexOAuthPanelProps {
  disabled?: boolean;
}

function formatExpiry(expiresAt: number | null | undefined, locale: string): string {
  if (!expiresAt) return '-';
  return new Date(expiresAt * 1000).toLocaleString(locale);
}

/**
 * Device-code authorization for the OpenAI-OAuth generation backend.
 *
 * The browser never sees token material: it shows the user code, then polls a
 * server-side session until the backend has stored the credential itself.
 */
export const CodexOAuthPanel: React.FC<CodexOAuthPanelProps> = ({ disabled = false }) => {
  const { t, language } = useUiLanguage();
  const locale = language === 'en' ? 'en-US' : 'zh-CN';

  const [status, setStatus] = useState<CodexOAuthStatus | null>(null);
  const [session, setSession] = useState<CodexOAuthLoginStart | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<ParsedApiError | null>(null);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      if (pollTimer.current) clearTimeout(pollTimer.current);
      if (copyTimer.current) clearTimeout(copyTimer.current);
    };
  }, []);

  const refreshStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const next = await systemConfigApi.getCodexOAuthStatus();
      if (isMounted.current) setStatus(next);
    } catch (err) {
      if (isMounted.current) setError(getParsedApiError(err));
    } finally {
      if (isMounted.current) setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  const pollSession = useCallback(
    async (sessionId: string, intervalSeconds: number) => {
      try {
        const next = await systemConfigApi.getCodexOAuthLoginSession(sessionId);
        if (!isMounted.current) return;

        if (next.state === 'pending') {
          pollTimer.current = setTimeout(
            () => void pollSession(sessionId, intervalSeconds),
            Math.max(1, intervalSeconds) * 1000,
          );
          return;
        }

        setSession(null);
        if (next.state === 'authorized') {
          await refreshStatus();
        } else if (next.state === 'failed') {
          setLoginError(next.message || next.reason || t('settings.codexOAuthLoginFailed'));
        } else if (next.state === 'unknown') {
          setLoginError(t('settings.codexOAuthSessionExpired'));
        }
      } catch (err) {
        if (isMounted.current) {
          setSession(null);
          setError(getParsedApiError(err));
        }
      }
    },
    [refreshStatus, t],
  );

  const handleStartLogin = useCallback(async () => {
    setIsStarting(true);
    setLoginError(null);
    setError(null);
    try {
      const started = await systemConfigApi.startCodexOAuthLogin();
      if (!isMounted.current) return;
      setSession(started);
      pollTimer.current = setTimeout(
        () => void pollSession(started.sessionId, started.intervalSeconds),
        Math.max(1, started.intervalSeconds) * 1000,
      );
    } catch (err) {
      if (isMounted.current) setError(getParsedApiError(err));
    } finally {
      if (isMounted.current) setIsStarting(false);
    }
  }, [pollSession]);

  const handleCancel = useCallback(async () => {
    if (!session) return;
    if (pollTimer.current) clearTimeout(pollTimer.current);
    const sessionId = session.sessionId;
    setSession(null);
    try {
      await systemConfigApi.cancelCodexOAuthLogin(sessionId);
    } catch {
      // Cancelling is best effort: the session expires on its own anyway.
    }
  }, [session]);

  const handleCopyCode = useCallback(async () => {
    if (!session) return;
    // navigator.clipboard is undefined on plain-HTTP LAN deployments (this app
    // is reverse-proxied over http://), so go through the shared helper: it
    // falls back to execCommand and reports whether the copy really happened.
    const succeeded = await copyToClipboard(session.userCode);
    setCopied(succeeded);
    setCopyFailed(!succeeded);
    if (copyTimer.current) clearTimeout(copyTimer.current);
    copyTimer.current = setTimeout(() => {
      if (!isMounted.current) return;
      setCopied(false);
      setCopyFailed(false);
    }, 2000);
  }, [session]);

  return (
    <div id={CODEX_OAUTH_PANEL_DOM_ID} className="rounded-xl border settings-border bg-background/35 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <KeyRound className="h-4 w-4 text-muted-text" aria-hidden="true" />
          <span className="text-sm font-semibold text-foreground">
            {t('settings.codexOAuthTitle')}
          </span>
          {status ? (
            <Badge variant={status.authorized ? 'success' : 'warning'} size="sm">
              {status.authorized
                ? t('settings.codexOAuthAuthorized')
                : t('settings.codexOAuthNotAuthorized')}
            </Badge>
          ) : null}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => void refreshStatus()}
          disabled={disabled || isLoading}
        >
          <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} aria-hidden="true" />
          {t('settings.codexOAuthRefresh')}
        </Button>
      </div>

      <p className="mt-2 text-xs leading-5 text-muted-text">
        {t('settings.codexOAuthDescription')}
      </p>

      {error ? <ApiErrorAlert error={error} className="mt-3" /> : null}

      {loginError ? (
        <SettingsAlert
          variant="warning"
          title={t('settings.codexOAuthLoginFailed')}
          message={loginError}
          className="mt-3"
        />
      ) : null}

      {status?.authorized ? (
        <div className="mt-3 space-y-1 text-xs text-muted-text">
          <div className="flex items-center gap-2 text-success">
            <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
            <span className="font-medium">{status.email}</span>
            {status.planType ? (
              <Badge variant="history" size="sm">
                {status.planType}
              </Badge>
            ) : null}
          </div>
          <div>
            {t('settings.codexOAuthExpiresAt')}: {formatExpiry(status.expiresAt, locale)}
            {status.refreshable ? ` · ${t('settings.codexOAuthAutoRefresh')}` : ''}
          </div>
          <div>
            {t('settings.codexOAuthCredentialPath')}: <code>{status.authFile}</code>
          </div>
        </div>
      ) : null}

      {session ? (
        <div className="mt-3 rounded-lg border settings-border bg-background/60 px-4 py-3">
          <p className="text-xs text-muted-text">{t('settings.codexOAuthStep1')}</p>
          <a
            href={session.verificationUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
          >
            {session.verificationUrl}
            <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
          </a>

          <p className="mt-3 text-xs text-muted-text">{t('settings.codexOAuthStep2')}</p>
          <div className="mt-1 flex items-center gap-2">
            <code className="rounded bg-background px-3 py-1.5 text-lg font-semibold tracking-widest text-foreground">
              {session.userCode}
            </code>
            <Button variant="ghost" size="sm" onClick={() => void handleCopyCode()}>
              <Copy className="h-4 w-4" aria-hidden="true" />
              {copied ? t('common.copied') : t('common.copy')}
            </Button>
          </div>
          {copyFailed ? (
            <p className="mt-1 text-xs text-warning">{t('settings.codexOAuthCopyFailed')}</p>
          ) : null}

          <div className="mt-3 flex items-center justify-between gap-2">
            <span className="inline-flex items-center gap-2 text-xs text-muted-text">
              <RefreshCw className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              {t('settings.codexOAuthWaiting')}
            </span>
            <Button variant="ghost" size="sm" onClick={() => void handleCancel()}>
              {t('common.cancel')}
            </Button>
          </div>
        </div>
      ) : (
        <div className="mt-3">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => void handleStartLogin()}
            disabled={disabled || isStarting}
          >
            <KeyRound className="h-4 w-4" aria-hidden="true" />
            {status?.authorized
              ? t('settings.codexOAuthReauthorize')
              : t('settings.codexOAuthAuthorize')}
          </Button>
        </div>
      )}
    </div>
  );
};
