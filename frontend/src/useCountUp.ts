import { useEffect, useState } from 'react';

const EASE_OUT = (t: number) => 1 - Math.pow(1 - t, 3);

/** Animates a number from 0 to `target` once, then holds. Skips straight to
 * the target under prefers-reduced-motion. Used for run-completion stats:
 * counting a real metric up is worth doing once, on the one screen that
 * reports it, not as a reusable list-view decoration. */
export function useCountUp(target: number | null | undefined, durationMs = 900): number | null {
  const [value, setValue] = useState<number | null>(target ?? null);

  useEffect(() => {
    if (target === null || target === undefined || Number.isNaN(target)) {
      setValue(null);
      return;
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setValue(target);
      return;
    }
    let frame: number;
    const start = performance.now();
    const from = 0;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / durationMs);
      setValue(from + (target - from) * EASE_OUT(progress));
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, durationMs]);

  return value;
}
