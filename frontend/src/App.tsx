import { Route, Routes } from 'react-router-dom';
import Home from './Home';
import Runner from './Runner';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/run/:runId" element={<Runner />} />
    </Routes>
  );
}

