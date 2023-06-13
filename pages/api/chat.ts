import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';
import { OpenAIModel } from '@/types/openai';

import GPT3Tokenizer from 'gpt3-tokenizer';

export const config = {
  runtime: 'edge',
};

const handler = async (req: Request): Promise<Response> => {
  try {
    const { model, messages, key, prompt, temperature } = (await req.json()) as ChatBody;

    const encoding = new GPT3Tokenizer({ type: 'gpt3' });

    let promptToSend = prompt;
    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    const MAX_TOKENS = process.env.MAX_TOKENS ? Number.parseInt(process.env.MAX_TOKENS) : 1000

    const prompt_tokens = encoding.encode(promptToSend).bpe;

    let tokenCount = prompt_tokens.length;
    let messagesToGoEvaluate: Message[] = [...messages];
    let messagesToSend: Message[] = [];

    const memoryStyle = process.env.MEMORY_STYLE

    switch(memoryStyle) {
      case 'includeFirstPrompt':
        messagesToSend = includeFirstMessageHandling(model, messagesToGoEvaluate, tokenCount, encoding, MAX_TOKENS);
        break;
      case 'default':
      default:
        messagesToSend = defaultHandling(model, messagesToGoEvaluate, tokenCount, encoding, MAX_TOKENS);
    }

    const stream = await OpenAIStream(model, promptToSend, temperatureToUse, key, messagesToSend);

    return new Response(stream);
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

const includeFirstMessageHandling = (model: OpenAIModel, messagesToGoEvaluate: Message[], tokenCount: number, encoding: GPT3Tokenizer, maxGenerationTokens: number): Message[] => {
  let messagesToSend: Message[] = [];

  if (messagesToGoEvaluate.length > 0) {
    const firstMessage = messagesToGoEvaluate[0];
    const firstTokens = encoding.encode(firstMessage.content).bpe;
    
    if (tokenCount + firstTokens.length + maxGenerationTokens <= model.tokenLimit) {
      tokenCount += firstTokens.length;

      for (let i = messagesToGoEvaluate.length - 1; i >= 1; i--) {
        const message = messagesToGoEvaluate[i];
        const tokens = encoding.encode(message.content).bpe;
  
        if (tokenCount + tokens.length + maxGenerationTokens > model.tokenLimit) {
          break;
        }
        tokenCount += tokens.length;
        messagesToSend = [message, ...messagesToSend];
      }

      messagesToSend = [firstMessage, ...messagesToSend]
    }
  }
  console.log(`Tokens: ${tokenCount}/${model.tokenLimit}`)
  return messagesToSend;
}

const defaultHandling = (model: OpenAIModel, messagesToGoEvaluate: Message[], tokenCount: number, encoding: GPT3Tokenizer, maxGenerationTokens: number): Message[] => {
  let messagesToSend: Message[] = [];

  for (let i = messagesToGoEvaluate.length - 1; i >= 0; i--) {
    const message = messagesToGoEvaluate[i];
    const tokens = encoding.encode(message.content).bpe;

    if (tokenCount + tokens.length + maxGenerationTokens > model.tokenLimit) {
      break;
    }
    tokenCount += tokens.length;
    messagesToSend = [message, ...messagesToSend];
  }
  console.log(`Tokens: ${tokenCount}/${model.tokenLimit}`)
  return messagesToSend;
}

export default handler;
