import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const { message } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // Path to our Python backend (parent directory)
    const pythonBackendPath = path.join(process.cwd(), '..');
    
    // Execute the Python script with the query as an argument
    const { stdout, stderr } = await execAsync(
      `cd "${pythonBackendPath}" && source venv/bin/activate && OPENAI_API_KEY="${process.env.OPENAI_API_KEY}" python3 web_query.py "${message.replace(/"/g, '\\"')}"`,
      { 
        timeout: 30000, // 30 second timeout
        env: { 
          ...process.env,
          PYTHONPATH: pythonBackendPath,
          OPENAI_API_KEY: process.env.OPENAI_API_KEY
        }
      }
    );

    if (stderr && !stderr.includes('UserWarning')) {
      console.error('Python stderr:', stderr);
    }

    try {
      const result = JSON.parse(stdout.trim());
      const response = result.response || "I'm having trouble right now. Try asking about my music taste!";
      return NextResponse.json({ response });
    } catch (parseError) {
      const response = stdout.trim() || "I'm having trouble right now. Try asking about my music taste!";
      return NextResponse.json({ response });
    }

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { 
        response: "Sorry, I'm having some technical difficulties right now. Try asking me about my music taste in a bit!" 
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ 
    message: "Chat API is running. Send a POST request with a message to chat with Akriti about her music taste." 
  });
}
