"""
DYCP 混合请求测试脚本

用法：
  python test_dycp_mixed.py --url http://localhost:8003 --model Qwen3-30B-A3B

功能：
  - 发送混合 batch：若干长请求（触发 DYCP CP 路径）+ 若干短请求（走普通路径）
  - 并发发送，观察长短请求能否同时正常完成
  - 打印每个请求的 TTFT、总延迟、输出 token 数
"""

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field

import aiohttp


TIMEOUT = aiohttp.ClientTimeout(total=3600)


@dataclass
class Result:
    req_id: str
    kind: str          # "long" or "short"
    prompt_tokens: int
    success: bool = False
    output_text: str = ""
    output_tokens: int = 0
    ttft: float = 0.0
    total_latency: float = 0.0
    error: str = ""


def make_long_prompt(target_tokens: int) -> str:
    """生成约 target_tokens 个 token 的 prompt（英文单词约 1.3 token/词）。"""
    unit = "The quick brown fox jumps over the lazy dog. " * 50
    repeat = max(1, target_tokens // (len(unit.split()) * 13 // 10))
    return (unit * repeat)[:target_tokens * 4]  # 粗略截断，tokenizer 会精确计数


def make_short_prompt() -> str:
    return "What is the capital of France? Answer in one sentence."


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    req_id: str,
    kind: str,
    prompt: str,
    max_tokens: int,
    prompt_tokens: int,
) -> Result:
    result = Result(req_id=req_id, kind=kind, prompt_tokens=prompt_tokens)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    first_token = False
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    if not first_token:
                        result.ttft = time.perf_counter() - t0
                        first_token = True
                    result.output_text += delta
                    result.output_tokens += 1
        result.success = True
    except Exception as e:
        result.error = str(e)
    result.total_latency = time.perf_counter() - t0
    return result


async def run(
    url: str,
    model: str,
    num_long: int,
    num_short: int,
    long_prompt_tokens: int,
    long_max_tokens: int,
    short_max_tokens: int,
) -> None:
    api_url = url.rstrip("/") + "/v1/chat/completions"

    tasks = []
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        entries = []
        for i in range(num_long):
            entries.append(("long", i))
        for i in range(num_short):
            entries.append(("short", i))
        random.shuffle(entries)

        for kind, i in entries:
            if kind == "long":
                prompt = make_long_prompt(long_prompt_tokens)
                tasks.append(
                    send_request(
                        session, api_url, model,
                        req_id=f"long-{i}",
                        kind="long",
                        prompt=prompt,
                        max_tokens=long_max_tokens,
                        prompt_tokens=long_prompt_tokens,
                    )
                )
            else:
                tasks.append(
                    send_request(
                        session, api_url, model,
                        req_id=f"short-{i}",
                        kind="short",
                        prompt=make_short_prompt(),
                        max_tokens=short_max_tokens,
                        prompt_tokens=20,
                    )
                )

        print(f"发送 {num_long} 个长请求（~{long_prompt_tokens} tokens）"
              f" + {num_short} 个短请求，并发执行...\n")
        t_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    # 打印结果
    print(f"{'ID':<12} {'类型':<6} {'prompt_tok':>10} {'output_tok':>10} "
          f"{'TTFT(s)':>9} {'总延迟(s)':>10} {'状态'}")
    print("-" * 75)
    for r in results:
        status = "OK" if r.success else f"FAIL: {r.error[:40]}"
        print(f"{r.req_id:<12} {r.kind:<6} {r.prompt_tokens:>10} "
              f"{r.output_tokens:>10} {r.ttft:>9.2f} "
              f"{r.total_latency:>10.2f}  {status}")

    print(f"\n全部请求完成，总耗时 {total_time:.2f}s")

    long_results = [r for r in results if r.kind == "long" and r.success]
    short_results = [r for r in results if r.kind == "short" and r.success]
    if long_results:
        avg_ttft = sum(r.ttft for r in long_results) / len(long_results)
        avg_lat  = sum(r.total_latency for r in long_results) / len(long_results)
        print(f"长请求  avg TTFT={avg_ttft:.2f}s  avg latency={avg_lat:.2f}s")
    if short_results:
        avg_ttft = sum(r.ttft for r in short_results) / len(short_results)
        avg_lat  = sum(r.total_latency for r in short_results) / len(short_results)
        print(f"短请求  avg TTFT={avg_ttft:.2f}s  avg latency={avg_lat:.2f}s")

    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n失败请求数: {len(failed)}")
        for r in failed:
            print(f"  {r.req_id}: {r.error}")


def main():
    parser = argparse.ArgumentParser(description="DYCP 混合请求测试")
    parser.add_argument("--url", default="http://localhost:8003",
                        help="vLLM server 地址")
    parser.add_argument("--model", default="Qwen3-30B-A3B",
                        help="模型名称（--served-model-name）")
    parser.add_argument("--num-long", type=int, default=2,
                        help="长请求数量")
    parser.add_argument("--num-short", type=int, default=4,
                        help="短请求数量")
    parser.add_argument("--long-prompt-tokens", type=int, default=10000,
                        help="长请求 prompt 约多少 token（需超过 --long-request-threshold）")
    parser.add_argument("--long-max-tokens", type=int, default=128,
                        help="长请求最大输出 token 数")
    parser.add_argument("--short-max-tokens", type=int, default=64,
                        help="短请求最大输出 token 数")
    args = parser.parse_args()

    asyncio.run(run(
        url=args.url,
        model=args.model,
        num_long=args.num_long,
        num_short=args.num_short,
        long_prompt_tokens=args.long_prompt_tokens,
        long_max_tokens=args.long_max_tokens,
        short_max_tokens=args.short_max_tokens,
    ))


if __name__ == "__main__":
    main()
