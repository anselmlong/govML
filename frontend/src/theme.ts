export type ThemeMode = 'light' | 'dark';

const KEY = 'govml-theme';

export function applyTheme(mode: ThemeMode): void {
  document.documentElement.dataset.theme = mode;
  document.documentElement.classList.toggle('dark', mode === 'dark');
  localStorage.setItem(KEY, mode);
}

export function initialTheme(): ThemeMode {
  const saved = localStorage.getItem(KEY);
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

