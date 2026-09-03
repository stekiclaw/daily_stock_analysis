const configuredApiBaseUrl = import.meta.env.VITE_API_URL?.trim();
const configuredAppBaseUrl = (import.meta.env.BASE_URL || '/').trim();

function normalizeAppBasePath(value: string): string {
  const path = value.replace(/^\/+|\/+$/g, '');
  return path ? `/${path}` : '/';
}

/** Router/static/API prefix baked in by Vite (for example `/dsa`). */
export const APP_BASE_PATH = normalizeAppBasePath(configuredAppBaseUrl);

/** Prefix an application-root path without duplicating the deployment base. */
export function withAppBasePath(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return APP_BASE_PATH === '/' ? normalizedPath : `${APP_BASE_PATH}${normalizedPath}`;
}

declare const __APP_PACKAGE_VERSION__: string | undefined;
declare const __APP_REVISION__: string | undefined;
declare const __APP_BUILD_TIME__: string | undefined;

const PLACEHOLDER_WEB_VERSION = '0.0.0';
const DEVELOPMENT_WEB_VERSION = 'development';
const UNKNOWN_REVISION = 'unknown';
const UNKNOWN_BUILD_TIME = '未提供';

// 默认保持同源 API，并沿用应用子路径（例如 /dsa），避免 POST 到根路径
// 后被 nginx 301 重定向。仅在显式提供 VITE_API_URL 时覆盖默认行为。
export const API_BASE_URL = configuredApiBaseUrl || (APP_BASE_PATH === '/' ? '' : APP_BASE_PATH);

export type WebBuildInfo = {
  version: string;
  rawVersion: string;
  revision: string;
  buildId: string;
  buildTime: string;
  isFallbackVersion: boolean;
};

function padBuildPart(value: number) {
  return value.toString().padStart(2, '0');
}

export function normalizeBuildTimestamp(buildTimestamp?: string) {
  const normalized = buildTimestamp?.trim();
  if (!normalized) {
    return UNKNOWN_BUILD_TIME;
  }

  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) {
    return normalized;
  }

  return parsed.toISOString();
}

export function createBuildIdentifier(buildTimestamp?: string) {
  const normalized = buildTimestamp?.trim();
  if (!normalized || normalized === UNKNOWN_BUILD_TIME) {
    return 'build-local';
  }

  const parsed = new Date(normalized);
  if (!Number.isNaN(parsed.getTime())) {
    const datePart = `${parsed.getUTCFullYear()}${padBuildPart(parsed.getUTCMonth() + 1)}${padBuildPart(parsed.getUTCDate())}`;
    const timePart = `${padBuildPart(parsed.getUTCHours())}${padBuildPart(parsed.getUTCMinutes())}${padBuildPart(parsed.getUTCSeconds())}`;
    return `build-${datePart}-${timePart}Z`;
  }

  const compactValue = normalized.replace(/[^0-9A-Za-z]+/g, '-').replace(/^-+|-+$/g, '');
  return compactValue ? `build-${compactValue}` : 'build-local';
}

export function resolveWebBuildInfo({
  packageVersion,
  revision,
  buildTimestamp,
}: {
  packageVersion?: string;
  revision?: string;
  buildTimestamp?: string;
}): WebBuildInfo {
  const rawVersion = packageVersion?.trim() || PLACEHOLDER_WEB_VERSION;
  const normalizedRevision = revision?.trim() || UNKNOWN_REVISION;
  const buildTime = normalizeBuildTimestamp(buildTimestamp);
  const buildId = createBuildIdentifier(buildTime);
  const isFallbackVersion = rawVersion === PLACEHOLDER_WEB_VERSION;

  return {
    version: isFallbackVersion ? DEVELOPMENT_WEB_VERSION : rawVersion,
    rawVersion,
    revision: normalizedRevision,
    buildId,
    buildTime,
    isFallbackVersion,
  };
}

const runtimePackageVersion = typeof __APP_PACKAGE_VERSION__ === 'string'
  ? __APP_PACKAGE_VERSION__.trim()
  : PLACEHOLDER_WEB_VERSION;
const runtimeRevision = typeof __APP_REVISION__ === 'string'
  ? __APP_REVISION__.trim()
  : UNKNOWN_REVISION;
const runtimeBuildTime = typeof __APP_BUILD_TIME__ === 'string'
  ? __APP_BUILD_TIME__.trim()
  : '';

export const WEB_BUILD_INFO = resolveWebBuildInfo({
  packageVersion: runtimePackageVersion,
  revision: runtimeRevision,
  buildTimestamp: runtimeBuildTime,
});
