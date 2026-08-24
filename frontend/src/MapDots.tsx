import { memo } from 'react';
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
  return (
    <>
      {nodes.map((node) => {
        const filtered = agencyFilter !== null && node.agency !== agencyFilter;
        const active = selectedId === node.dataset_id;
        const matched = matchedIds.has(node.dataset_id);
        return (
          <circle
            key={node.dataset_id}
            className={`map-dot ${active ? 'active' : ''} ${matched ? 'matched' : ''}`}
            cx={node.x * 940 + 30}
            cy={node.y * 640 + 30}
            r={active ? 5.5 : matched ? 4 : 2.2}
            fill={colorFor.get(node.dataset_id) ?? '#737373'}
            opacity={filtered ? 0.08 : matched || active || !hasResults ? 0.86 : 0.24}
            onClick={() => onSelect(node)}
          />
        );
      })}
    </>
  );
}

export default memo(MapDots);
