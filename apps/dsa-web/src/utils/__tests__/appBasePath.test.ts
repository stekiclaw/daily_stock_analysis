import { describe, expect, it, vi } from 'vitest'

/**
 * DSA is served both at `/` and behind nginx at `/dsa/` (which strips the
 * prefix before proxying). Everything the browser resolves itself — router,
 * static index, login redirect, API base — must carry the prefix, or a POST to
 * `/api/...` gets 301'd to `/dsa/api/...` and loses its method and body.
 */
const loadConstants = async (baseUrl: string) => {
  vi.resetModules()
  vi.stubEnv('BASE_URL', baseUrl)
  try {
    return await import('../constants')
  } finally {
    vi.unstubAllEnvs()
  }
}

describe('APP_BASE_PATH', () => {
  it('stays at the root when no base path is configured', async () => {
    const { APP_BASE_PATH, API_BASE_URL, withAppBasePath } = await loadConstants('/')

    expect(APP_BASE_PATH).toBe('/')
    expect(API_BASE_URL).toBe('')
    expect(withAppBasePath('/login')).toBe('/login')
    expect(withAppBasePath('/stocks.index.json')).toBe('/stocks.index.json')
  })

  it('normalizes the Vite base and prefixes browser-resolved paths', async () => {
    const { APP_BASE_PATH, API_BASE_URL, withAppBasePath } = await loadConstants('/dsa/')

    expect(APP_BASE_PATH).toBe('/dsa')
    // Requests go to /dsa/api/...; nginx strips /dsa/ before FastAPI sees them.
    expect(API_BASE_URL).toBe('/dsa')
    expect(withAppBasePath('/login')).toBe('/dsa/login')
    expect(withAppBasePath('stocks.index.json')).toBe('/dsa/stocks.index.json')
  })

  it('tolerates a base path written without surrounding slashes', async () => {
    const { APP_BASE_PATH, withAppBasePath } = await loadConstants('dsa')

    expect(APP_BASE_PATH).toBe('/dsa')
    expect(withAppBasePath('/login')).toBe('/dsa/login')
  })
})
