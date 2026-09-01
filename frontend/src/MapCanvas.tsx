import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { select, zoom, zoomIdentity, ZoomBehavior, ZoomTransform } from 'd3';
import { MapDataset } from './api';

export interface MapCanvasHandle {
  flyTo: (node: MapDataset) => void;
  resetView: () => void;
}

interface MapCanvasProps {
  nodes: MapDataset[];
  colorFor: Map<string, string>;
  agencyFilter: string | null;
  selectedId?: string;
  matchedIds: Set<string>;
  hasResults: boolean;
  onSelect: (node: MapDataset) => void;
}

// design space matches the old SVG viewBox so zoom math feels identical
const DESIGN_W = 1000;
const DESIGN_H = 700;

interface Phase {
  phi: number;
  w: number;
  amp: number;
  stag: number;
  hue: number;
}

function MapCanvas(
  { nodes, colorFor, agencyFilter, selectedId, matchedIds, hasResults, onSelect }: MapCanvasProps,
  ref: React.Ref<MapCanvasHandle>
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ttRef = useRef<HTMLDivElement | null>(null);
  const behaviorRef = useRef<ZoomBehavior<HTMLCanvasElement, unknown> | null>(null);
  const tfRef = useRef<ZoomTransform>(zoomIdentity);
  const mountRef = useRef<number>(performance.now());
  const movedRef = useRef(false);
  const downPosRef = useRef<{ x: number; y: number } | null>(null);
  const hoverIdRef = useRef<string | null>(null);
  const [tooltip, setTooltip] = useState<{ name: string; agency: string; x: number; y: number } | null>(null);

  // keep latest props reachable from the rAF loop without resubscribing zoom
  const propsRef = useRef({ nodes, colorFor, agencyFilter, selectedId, matchedIds, hasResults, onSelect });
  propsRef.current = { nodes, colorFor, agencyFilter, selectedId, matchedIds, hasResults, onSelect };

  // stable per-dot motion signature: phase, speed, amplitude, stagger, wobble
  const phases = useMemo(() => {
    const map = new Map<string, Phase>();
    for (const node of nodes) {
      let h = 7;
      for (let i = 0; i < node.dataset_id.length; i++) h = (h * 31 + node.dataset_id.charCodeAt(i)) | 0;
      const r1 = ((h >>> 8) & 0xffff) / 0xffff;
      const r2 = ((h >>> 16) & 0xffff) / 0xffff;
      const r3 = ((h >>> 24) & 0xff) / 0xff;
      map.set(node.dataset_id, {
        phi: r1 * Math.PI * 2,
        w: 0.25 + r2 * 0.45, // angular velocity rad/s
        amp: 1.6 + r3 * 2.2, // drift radius in design units (≈2–4 screen px at fit)
        stag: ((h >>> 2) & 0xff) / 0xff, // entrance stagger 0..1
        hue: r2
      });
    }
    return map;
  }, [nodes]);

  const fitTransform = useCallback((): ZoomTransform => {
    const canvas = canvasRef.current;
    if (!canvas) return zoomIdentity;
    const k = Math.min(canvas.clientWidth / DESIGN_W, canvas.clientHeight / DESIGN_H);
    return zoomIdentity.translate(
      (canvas.clientWidth - DESIGN_W * k) / 2,
      (canvas.clientHeight - DESIGN_H * k) / 2
    ).scale(k);
  }, []);

  const draw = useCallback(
    (time: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const { nodes: n, colorFor: colors, agencyFilter: filter, selectedId: sel, matchedIds: matched, hasResults, onSelect: _onSelect } =
        propsRef.current;

      const dpr = window.devicePixelRatio || 1;
      const W = canvas.clientWidth;
      const H = canvas.clientHeight;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const tf = tfRef.current;
      ctx.translate(tf.x, tf.y);
      ctx.scale(tf.k, tf.k);

      const cs = getComputedStyle(canvas);
      const bearing = cs.getPropertyValue('--bearing').trim() || '#2f6e7f';
      const depth = cs.getPropertyValue('--depth').trim() || '#333';
      const t = time * 0.001; // seconds
      const elapsed = (time - mountRef.current) * 0.001;

      for (let i = 0; i < n.length; i++) {
        const node = n[i];
        const ph = phases.get(node.dataset_id);
        const phi = ph ? ph.phi : i * 0.37;
        const w = ph ? ph.w : 0.4;
        const amp = ph ? ph.amp : 2;

        const filtered = filter !== null && node.agency !== filter;
        const active = sel === node.dataset_id;
        const isMatched = matched.has(node.dataset_id);

        // ambient drift: lissajous-ish wander, clearly visible at any zoom
        const ox = Math.sin(t * w + phi) * amp;
        const oy = Math.cos(t * w * 0.83 + phi * 1.7) * amp;
        const cx = node.x * 940 + 30 + ox;
        const cy = node.y * 640 + 30 + oy;

        // gentle size breathing
        let r = (isMatched ? 3.2 : 2.2) * (1 + 0.14 * Math.sin(t * w * 1.3 + phi * 2.1));
        if (isMatched) r += 0.55 * Math.sin(t * 3.4 + phi); // search pulse
        // hover/selection never inflate the dot — ring + full opacity carry the emphasis
        const emphasized = active || hoverIdRef.current === node.dataset_id;

        // opacity semantics preserved from the SVG version
        let alpha = filtered ? 0.08 : isMatched || active || !hasResults ? 0.86 : 0.24;
        // entrance cascade over the first ~1.6s after mount
        const appear = Math.min(1, Math.max(0, (elapsed - (ph ? ph.stag * 0.9 : 0)) / 0.5));
        alpha *= appear;

        if (alpha <= 0.01) continue;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = colors.get(node.dataset_id) ?? '#737373';
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(r, 0.05), 0, Math.PI * 2);
        ctx.fill();

        if (active || isMatched || emphasized) {
          ctx.globalAlpha = Math.min(1, alpha + 0.14);
          // ring drawn in screen-space width so it stays crisp at any zoom
          ctx.lineWidth = 2 / tf.k;
          ctx.strokeStyle = active ? bearing : emphasized ? bearing : depth;
          ctx.stroke();
        }
      }

      ctx.globalAlpha = 1;
    },
    [phases]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    mountRef.current = performance.now();
    movedRef.current = false;
    tfRef.current = fitTransform();

    const behavior = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.6, 22])
      .translateExtent([
        [0, 0],
        [DESIGN_W, DESIGN_H]
      ])
      .on('start', () => {
        movedRef.current = true;
        canvas.style.cursor = 'grabbing';
      })
      .on('zoom', (event) => {
        tfRef.current = event.transform;
      })
      .on('end', () => {
        canvas.style.cursor = hoverIdRef.current ? "pointer" : "grab";
      });
    select(canvas).call(behavior).on('dblclick.zoom', null);
    behaviorRef.current = behavior;

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    if (reduceMotion) {
      draw(performance.now());
    } else {
      const loop = (t: number) => {
        draw(t);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }

    const ro = new ResizeObserver(() => {
      if (!movedRef.current) tfRef.current = fitTransform();
      if (reduceMotion) draw(performance.now());
    });
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      select(canvas).on('.zoom', null);
      behaviorRef.current = null;
    };
  }, [draw, fitTransform]);

  // hover cursor + suppression of click-after-drag
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const toUser = (clientX: number, clientY: number) => {
      const rect = canvas.getBoundingClientRect();
      const tf = tfRef.current;
      return { x: (clientX - rect.left - tf.x) / tf.k, y: (clientY - rect.top - tf.y) / tf.k };
    };

    const pick = (ux: number, uy: number): MapDataset | null => {
      const { nodes: n } = propsRef.current;
      const tf = tfRef.current;
      const slack = 6 / tf.k;
      let best: MapDataset | null = null;
      let bestD2 = Infinity;
      for (const node of n) {
        const cx = node.x * 940 + 30;
        const cy = node.y * 640 + 30;
        const d2 = (cx - ux) * (cx - ux) + (cy - uy) * (cy - uy);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = node;
        }
      }
      return bestD2 <= slack * slack ? best : null;
    };

    const onMove = (event: PointerEvent) => {
      const { x, y } = toUser(event.clientX, event.clientY);
      const hit = pick(x, y);
      hoverIdRef.current = hit?.dataset_id ?? null;
      canvas.style.cursor = hit ? 'pointer' : 'grab';
      if (hit) {
        const rect = canvas.getBoundingClientRect();
        setTooltip({
          name: hit.name,
          agency: hit.agency || '',
          // position near the cursor, in page coords (tooltip is fixed)
          x: event.clientX - rect.left,
          y: event.clientY - rect.top
        });
      } else {
        setTooltip(null);
      }
    };

    const onDown = (event: PointerEvent) => {
      downPosRef.current = { x: event.clientX, y: event.clientY };
    };

    const onClick = (event: MouseEvent) => {
      const down = downPosRef.current;
      if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) > 4) return; // was a drag
      const { x, y } = toUser(event.clientX, event.clientY);
      const hit = pick(x, y);
      if (hit) propsRef.current.onSelect(hit);
    };

    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('click', onClick);
    return () => {
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('click', onClick);
    };
  }, []);

  useImperativeHandle(
    ref,
    () => ({
      resetView: () => {
        const canvas = canvasRef.current;
        if (!canvas || !behaviorRef.current) return;
        movedRef.current = false;
        select(canvas)
          .transition()
          .duration(360)
          .call(behaviorRef.current.transform, fitTransform());
      },
      flyTo: (node: MapDataset) => {
        const canvas = canvasRef.current;
        if (!canvas || !behaviorRef.current) return;
        movedRef.current = true;
        const cx = node.x * 940 + 30;
        const cy = node.y * 640 + 30;
        select(canvas)
          .transition()
          .duration(650)
          .ease((t: number) => 1 - Math.pow(1 - t, 3))
          .call(
            behaviorRef.current.transform,
            zoomIdentity.translate(canvas.clientWidth / 2, canvas.clientHeight / 2).scale(6).translate(-cx, -cy)
          );
      }
    }),
    [fitTransform]
  );

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Semantic map of Singapore open datasets"
        style={{ display: 'block', width: '100%', height: '100%', touchAction: 'none', cursor: 'grab' }}
      />
      {tooltip && (
        <div
          className="map-tooltip"
          style={{
            position: 'absolute',
            left: Math.min(tooltip.x + 14, (canvasRef.current?.clientWidth ?? 9999) - 240),
            top: tooltip.y + 12,
            pointerEvents: 'none'
          }}
        >
          <b>{tooltip.name}</b>
          {tooltip.agency && <small>{tooltip.agency}</small>}
        </div>
      )}
    </div>
  );
}

export default forwardRef(MapCanvas);
