'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'akriti';
  timestamp: Date;
}

export default function ChatWithAkriti() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    setMessages([
      {
        id: '1',
        content: "Hey! I'm Akriti 👋 Ask me anything about my music taste - what genres I listen to, my favorite artists, cultural connections in my collection, or anything else about my musical preferences!",
        sender: 'akriti',
        timestamp: new Date()
      }
    ]);
  }, []);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputValue }),
      });

      const data = await response.json();

      const akritiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response || "Sorry, I couldn't process that right now. Try asking about my music taste!",
        sender: 'akriti',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, akritiMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Oops! Something went wrong. Try asking about my music taste again!",
        sender: 'akriti',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-black font-['Manrope',sans-serif] flex justify-center">
      <div className="w-full max-w-4xl bg-white shadow-sm flex flex-col min-h-screen">
        {/* Header */}
        <div className="border-b border-gray-200 p-6 text-center">
          <h1 className="text-2xl font-medium">Chat with Akriti</h1>
          <p className="text-sm text-gray-600 mt-2">Ask me about my music taste • Graph RAG vs Naive RAG comparison</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] p-4 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-black text-white'
                  : 'bg-gray-50 text-black border border-gray-200'
              }`}
            >
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.sender === 'akriti' ? (
                  <div dangerouslySetInnerHTML={{
                    __html: message.content
                      .replace(/🕸️ Processing query with complete Graph RAG pipeline\.\.\./g, '<div class="bg-blue-50 p-3 rounded mb-4 border-l-4 border-blue-500"><strong>🔍 Processing Query</strong></div>')
                      .replace(/📍 Extracted entities: (.+)/g, '<div class="text-sm text-gray-600 mb-1">• Found entities: <strong>$1</strong></div>')
                      .replace(/🔍 Found (\d+) candidate relations from subgraph expansion/g, '<div class="text-sm text-gray-600 mb-1">• Graph expansion: <strong>$1 relations</strong></div>')
                      .replace(/🧠 LLM reranked to (\d+) most relevant relations/g, '<div class="text-sm text-gray-600 mb-4">• Reranked to top <strong>$1</strong></div>')
                      .replace(/={60}/g, '<div class="border-t border-gray-200 my-6"></div>')
                      .replace(/📊 COMPARISON: Graph RAG vs Naive RAG/g, '<h3 class="text-lg font-semibold mb-4 pb-2 border-b">📊 Results Comparison</h3>')
                      .replace(/📋 Passages from Naive RAG:/g, '<div class="bg-gray-50 p-4 rounded-lg mb-4"><h4 class="font-medium text-gray-700 mb-3">📋 Naive RAG</h4>')
                      .replace(/🕸️ Passages from Graph RAG:/g, '</div><div class="bg-blue-50 p-4 rounded-lg mb-4"><h4 class="font-medium text-blue-700 mb-3">🕸️ Graph RAG</h4>')
                      .replace(/🔍 \*\*Naive RAG Answer:\*\*/g, '</div><div class="bg-white border-l-4 border-gray-400 p-4 mb-4"><h4 class="font-semibold text-gray-700 mb-3">Naive RAG</h4>')
                      .replace(/🕸️ \*\*Graph RAG Answer:\*\*/g, '</div><div class="bg-white border-l-4 border-blue-500 p-4 mb-4"><h4 class="font-semibold text-blue-700 mb-3">Graph RAG</h4>')
                      .replace(/📈 \*\*Analysis:\*\*/g, '</div><div class="bg-green-50 p-4 rounded-lg border border-green-200"><h4 class="font-semibold text-green-700 mb-2">Why Graph RAG?</h4>')
                      .replace(/\[Naive RAG\]/g, '')
                      .replace(/\[Graph RAG\]/g, '')
                      + '</div>'
                  }} />
                ) : (
                  <div className="font-mono">{message.content}</div>
                )}
              </div>
              {isClient && (
                <p className={`text-xs mt-2 ${
                  message.sender === 'user' ? 'text-gray-300' : 'text-gray-500'
                }`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-black border border-gray-200 p-3 rounded-lg">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

        {/* Input */}
        <div className="border-t border-gray-200 p-6">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about my music taste..."
            className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="px-6 py-3 bg-black text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
        
        {/* Example prompts */}
        <div className="mt-3 flex flex-wrap gap-2">
          {[
            "What genres do you listen to?",
            "Who are your favorite rock artists?",
            "What indie music do you listen to?",
            "How do your music tastes connect?"
          ].map((prompt) => (
            <button
              key={prompt}
              onClick={() => setInputValue(prompt)}
              className="text-xs px-3 py-1 bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
              disabled={isLoading}
            >
              {prompt}
            </button>
          ))}
        </div>
        </div>
      </div>
    </div>
  );
}