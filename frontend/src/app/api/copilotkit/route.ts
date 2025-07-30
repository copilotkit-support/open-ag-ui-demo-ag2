import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
  OpenAIAdapter
} from '@copilotkit/runtime';
import { NextRequest } from 'next/server';
import { HttpAgent } from "@ag-ui/client";


const ag2Agent = new HttpAgent({
  // url: process.env.NEXT_PUBLIC_AG2_URL || "http://0.0.0.0:8000/ag2-agent",
  url: "http://0.0.0.0:8000/ag2-agent",
});
const serviceAdapter = new OpenAIAdapter()
const runtime = new CopilotRuntime({
  agents: {
    // @ts-ignore
    ag2Agent : ag2Agent 
  },
});
// const runtime = new CopilotRuntime()
export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: '/api/copilotkit',
  });

  return handleRequest(req);
};