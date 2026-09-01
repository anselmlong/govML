import { memo, useMemo } from 'react';
import { MapDataset } from './api';

interface MapDotsProps {
  nodes: MapDataset[];
  colorFor: Map<string, string>;
  agencyFilter: string | null;
  selectedId?: string;
  matchedIds: Set<string>;
  hasResults: boolean;
  onSelect: (node: MapDataset) => void;
}

function MapDots({ nodes, colorFor, agencyFilter, selectedId, matchedIds, hasResults, onSelect }: MapDotsProps) {
  // stable per-dot phase offsets + durations so the ambient drift doesn't
  // look like a synchronized school of fish. derived from dataset_id hash.
  const phases = useMemo(() => {
    const map = new Map<string, { delay: number; dur: number; dx: number; dy: number }>();
    for (const node of nodes) {
      let h = 7;
      for (let i = 0; i < node.dataset_id.length; i++) h = (h * 31 + node.dataset_id.charCodeAt(i)) | 0;
      const r1 = ((h >>> 8) & 0xffff) / 0xffff;
      const r2 = ((h >>> 16) & 0xffff) / 0xffff;
      const r3 = ((h >>> 24) & 0xff) / 0xff;
      map.set(node.dataset_id, {
        delay: -r1 * 9,          // negative → mid-animation at load
        dur: 6 + r2 * 6,         // 6–12s per cycle
        dx: (r3 - 0.5) * 5,      // drift radius ±2.5px in viewBox units
        dy: (((h >>> 4) & 0xff) / 0xff - 0.5) * 5
      });
    }
    return map;
  }, [nodes]);

  return (
    <>
      {nodes.map((node, index) => {
        const filtered = agencyFilter !== null && node.agency !== agencyFilter;
        const active = selectedId === node.dataset_id;
        const matched = matchedIds.has(node.dataset_id);
        const ph = phases.get(node.dataset_id);
        return (
          <circle
            key={node.dataset_id}
            className={`map-dot ${active ? 'active' : ''} ${matched ? 'matched' : ''}`}
            cx={node.x * 940 + 30}
            cy={node.y * 640 + 30}
            r={active ? 5.5 : matched ? 4 : 2.2}
            fill={colorFor.get(node.dataset_id) ?? '#737373'}
            opacity={filtered ? 0.08 : matched || active || !hasResults ? 0.86 : 0.24}
            style={{
              // ambient life: each dot breathes and drifts on its own clock
              ...(ph
                ? {
                    animationDelay: `${ph.delay}s`,
                    animationDuration: `${ph.dur}s`,
                    '--dx': `${ph.dx}px`,
                    '--dy': `${ph.dy}px`
                  }
                : {}),
              // entrance cascade: tiny stagger by index, only meaningful on
              // first mount since keys are stable afterwards
              transitionDelay: `${Math.min(index * 0.15, 900)}ms`
            }}
            onClick={() => onSelect(node)}
          />
        );
      })}
    </>
  );
}

export default memo(MapDots);
