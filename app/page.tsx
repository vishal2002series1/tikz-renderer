"use client";

import { useState } from "react";
import Editor from "@monaco-editor/react";

const DEFAULT_CODE = `\\documentclass[varwidth=21cm, border=5mm]{standalone}
\\usepackage{tikz-3dplot}
\\begin{document}
\\begin{tikzpicture}[tdplot_main_coords, >=stealth, scale=2]
  \\draw[thick, ->] (0,0,0) -- (3,0,0) node[anchor=north east]{$x$};
  \\draw[thick, ->] (0,0,0) -- (0,3,0) node[anchor=north west]{$y$};
  \\draw[thick, ->] (0,0,0) -- (0,0,3) node[anchor=south]{$z$};
  \\shade[ball color=blue!20, opacity=0.8] (0,0,0) circle (1.5cm);
\\end{tikzpicture}
\\end{document}`;

export default function Home() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [svgContent, setSvgContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRender = async () => {
    setLoading(true);
    setError(null);
    setSvgContent(null);

    try {
      const response = await fetch("/api/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Failed to render");
      }

      const svgText = await response.text();
      setSvgContent(svgText);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- NEW: Download Handlers ---

  const handleDownloadSVG = () => {
    if (!svgContent) return;
    const blob = new Blob([svgContent], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "diagram.svg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadPNG = () => {
    if (!svgContent) return;
    
    const blob = new Blob([svgContent], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    
    img.onload = () => {
      const canvas = document.createElement("canvas");
      // Scale by 3 for a crisp, high-resolution PNG
      const scale = 3; 
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      // Fill with a white background (SVGs are transparent by default)
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      
      const pngUrl = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = pngUrl;
      link.download = "diagram.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  return (
    <main className="min-h-screen bg-gray-50 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">TikZ & LaTeX Renderer</h1>
            <p className="text-gray-500 mt-1">Compile and preview complex diagrams instantly.</p>
          </div>
          <button
            onClick={handleRender}
            disabled={loading}
            className="px-8 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Rendering..." : "Render Diagram"}
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Editor Section */}
          <div className="flex flex-col rounded-lg overflow-hidden shadow border border-gray-800 bg-[#1e1e1e]">
            <div className="p-2 bg-gray-800 text-gray-400 text-xs font-mono flex justify-between items-center">
              <span>editor.tex</span>
              <span>LaTeX</span>
            </div>
            <Editor
              height="750px"
              defaultLanguage="latex"
              theme="vs-dark"
              value={code}
              onChange={(value) => setCode(value || "")}
              options={{
                minimap: { enabled: false },
                wordWrap: "on",
                fontSize: 14,
                padding: { top: 16 },
              }}
            />
          </div>

          {/* Output Section */}
          <div className="flex flex-col bg-white rounded-lg shadow border border-gray-200 overflow-hidden min-h-[750px]">
            <div className="p-4 bg-gray-100 border-b border-gray-200 flex justify-between items-center">
              <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wider">Output</h2>
              
              {/* NEW: Download Buttons (Only visible on success) */}
              {svgContent && !loading && !error && (
                <div className="flex space-x-2">
                  <button 
                    onClick={handleDownloadSVG}
                    className="px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-300 rounded hover:bg-gray-50 transition-colors"
                  >
                    SVG
                  </button>
                  <button 
                    onClick={handleDownloadPNG}
                    className="px-3 py-1.5 text-xs font-medium text-white bg-green-600 border border-green-700 rounded hover:bg-green-700 transition-colors"
                  >
                    PNG
                  </button>
                </div>
              )}
            </div>
            
            <div className="flex-grow p-4 flex items-center justify-center overflow-auto bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] relative">
              {loading && (
                <div className="text-blue-600 animate-pulse font-medium">Compiling LaTeX Engine...</div>
              )}
              
              {error && (
                <div className="text-red-600 bg-red-50 p-4 rounded-md font-mono text-sm whitespace-pre-wrap w-full overflow-x-auto border border-red-200">
                  {error}
                </div>
              )}

              {!loading && !error && svgContent && (
                <div 
                  className="max-w-full max-h-full transition-opacity duration-300"
                  dangerouslySetInnerHTML={{ __html: svgContent }} 
                />
              )}

              {!loading && !error && !svgContent && (
                <div className="text-gray-400">Write some TikZ code and click Render.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}