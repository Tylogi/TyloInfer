import { expect, it } from 'vitest';
import type { Message } from '@shared/types/api';
import { textParts } from './messageParts';

it('消息正文与 reasoning 独立合并，不把思考混入可编辑正文', () => {
  const message = { parts: [
    { type: 'reasoning', text: '思考一' },
    { type: 'text', text: '正文' },
    { type: 'reasoning', text: '思考二' },
    { type: 'transcript', text: '转录' },
  ] } as Message;
  expect(textParts(message)).toEqual({ text: '正文转录', reasoning: '思考一思考二' });
});
