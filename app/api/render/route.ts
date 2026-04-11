import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';

const execAsync = promisify(exec);

export async function POST(req: NextRequest) {
  let tempDir: string | null = null;

  try {
    const { code } = await req.json();

    if (!code) {
      return NextResponse.json({ error: 'No LaTeX code provided' }, { status: 400 });
    }

    // 1. Create a secure temporary directory
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'tikz-render-'));
    const texPath = path.join(tempDir, 'diagram.tex');
    const pdfPath = path.join(tempDir, 'diagram.pdf');
    const svgPath = path.join(tempDir, 'diagram.svg');

    // 2. Write the LaTeX code to a .tex file
    await fs.writeFile(texPath, code, 'utf8');

    // 3. Compile LaTeX to PDF using pdflatex
    // -interaction=nonstopmode ensures the process doesn't hang waiting for user input on errors
    try {
      await execAsync(`pdflatex -interaction=nonstopmode -output-directory="${tempDir}" "${texPath}"`);
    } catch (error: any) {
      // If pdflatex fails, we read the log to see why
      const logPath = path.join(tempDir, 'diagram.log');
      let logOutput = error.message;
      try {
        logOutput = await fs.readFile(logPath, 'utf8');
      } catch (e) {
        // Log file might not exist if it failed before starting
      }
      throw new Error(`LaTeX compilation failed:\n${logOutput}`);
    }

    // 4. Convert the generated PDF to an SVG
    try {
      await execAsync(`pdf2svg "${pdfPath}" "${svgPath}"`);
    } catch (error: any) {
      throw new Error(`SVG conversion failed: ${error.message}`);
    }

    // 5. Read the SVG file and return it
    const svgContent = await fs.readFile(svgPath, 'utf8');
    
    return new NextResponse(svgContent, {
      headers: {
        'Content-Type': 'image/svg+xml',
        'Cache-Control': 'no-cache'
      }
    });

  } catch (error: any) {
    console.error('Rendering error:', error);
    return NextResponse.json(
      { error: error.message || 'An unknown error occurred during rendering.' },
      { status: 500 }
    );
  } finally {
    // 6. Cleanup: Always delete the temporary directory
    if (tempDir) {
      try {
        await fs.rm(tempDir, { recursive: true, force: true });
      } catch (cleanupError) {
        console.error('Failed to clean up temp directory:', cleanupError);
      }
    }
  }
}