import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { copyToClipboard } from '../clipboard'

describe('copyToClipboard', () => {
  const originalClipboard = navigator.clipboard
  const originalExecCommand = document.execCommand

  const setClipboard = (value: unknown) => {
    Object.defineProperty(navigator, 'clipboard', { value, configurable: true, writable: true })
  }

  beforeEach(() => {
    document.body.innerHTML = ''
  })

  afterEach(() => {
    setClipboard(originalClipboard)
    document.execCommand = originalExecCommand
    vi.restoreAllMocks()
  })

  it('uses the async Clipboard API when it is available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    setClipboard({ writeText })
    const execCommand = vi.fn()
    document.execCommand = execCommand

    await expect(copyToClipboard('report body')).resolves.toBe(true)
    expect(writeText).toHaveBeenCalledWith('report body')
    expect(execCommand).not.toHaveBeenCalled()
  })

  it('falls back to execCommand when the Clipboard API rejects', async () => {
    // Plain-HTTP LAN deployments have no secure context, so writeText throws.
    setClipboard({ writeText: vi.fn().mockRejectedValue(new Error('not allowed')) })
    let selectedText = ''
    document.execCommand = vi.fn(() => {
      selectedText = (document.activeElement as HTMLTextAreaElement)?.value ?? ''
      return true
    })

    await expect(copyToClipboard('report body')).resolves.toBe(true)
    expect(selectedText).toBe('report body')
  })

  it('falls back when the Clipboard API is missing entirely', async () => {
    setClipboard(undefined)
    document.execCommand = vi.fn(() => true)

    await expect(copyToClipboard('report body')).resolves.toBe(true)
    expect(document.execCommand).toHaveBeenCalledWith('copy')
  })

  it('reports failure when both paths fail, so no false "copied" state is shown', async () => {
    setClipboard({ writeText: vi.fn().mockRejectedValue(new Error('denied')) })
    document.execCommand = vi.fn(() => false)

    await expect(copyToClipboard('report body')).resolves.toBe(false)
  })

  it('reports failure when execCommand itself throws', async () => {
    setClipboard(undefined)
    document.execCommand = vi.fn(() => {
      throw new Error('unsupported')
    })

    await expect(copyToClipboard('report body')).resolves.toBe(false)
  })

  it('removes the scratch textarea and restores focus', async () => {
    setClipboard(undefined)
    const input = document.createElement('input')
    document.body.appendChild(input)
    input.focus()
    document.execCommand = vi.fn(() => true)

    await copyToClipboard('report body')

    expect(document.querySelector('textarea')).toBeNull()
    expect(document.activeElement).toBe(input)
  })

  it('rejects non-string input instead of copying "undefined"', async () => {
    setClipboard({ writeText: vi.fn() })
    await expect(copyToClipboard(undefined as unknown as string)).resolves.toBe(false)
  })
})
