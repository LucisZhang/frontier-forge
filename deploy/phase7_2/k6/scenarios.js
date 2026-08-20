import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";

const status200 = new Counter("forge_http_200");
const status429 = new Counter("forge_http_429");
const status500 = new Counter("forge_http_500");
const status503 = new Counter("forge_http_503");
const retryAfter429 = new Counter("forge_retry_after_429");
const requestLatency = new Trend("forge_request_latency", true);

const scenario = __ENV.SCENARIO || "steady";
const duration = __ENV.DURATION || "3m";
const rate = Number(__ENV.RATE || (scenario === "saturation" ? 30 : 12));
const gateway = __ENV.GATEWAY_URL || "http://forge-gateway.forge-system.svc:9000";

const executors = {
  steady: {
    executor: "constant-arrival-rate",
    duration,
    preAllocatedVUs: 16,
    maxVUs: 64,
    rate: Number(__ENV.RATE || 1),
    timeUnit: "1s",
  },
  autoscale: {
    executor: "ramping-arrival-rate",
    startRate: 1,
    timeUnit: "1s",
    preAllocatedVUs: 64,
    maxVUs: 256,
    stages: [
      { duration: "20s", target: rate },
      { duration, target: rate },
      { duration: "20s", target: 1 },
    ],
  },
  saturation: {
    executor: "constant-arrival-rate",
    duration,
    preAllocatedVUs: 128,
    maxVUs: 512,
    rate,
    timeUnit: "1s",
  },
  fault: {
    executor: "constant-arrival-rate",
    duration,
    preAllocatedVUs: 16,
    maxVUs: 64,
    rate: Number(__ENV.RATE || 2),
    timeUnit: "1s",
  },
};

export const options = {
  discardResponseBodies: false,
  scenarios: { [scenario]: executors[scenario] },
  thresholds: {
    checks: ["rate>0.90"],
  },
};

const body = JSON.stringify({
  model: "forge-r1b",
  messages: [
    {
      role: "user",
      content:
        "Return one compact JSON object with company, issue, product, urgency, ambiguity_flag, and tool_call.",
    },
  ],
  max_tokens: 96,
  temperature: 0,
  stream: false,
});

export default function () {
  const response = http.post(`${gateway}/v1/chat/completions`, body, {
    headers: { "Content-Type": "application/json", "X-Allow-Degrade": "0" },
    timeout: "70s",
    tags: { phase: "7.2", scenario },
  });
  check(response, {
    "expected HTTP contract": (item) =>
      item.status === 200 || item.status === 429 || item.status === 500 || item.status === 503,
    "429 includes Retry-After": (item) => item.status !== 429 || Boolean(item.headers["Retry-After"]),
  });
  requestLatency.add(response.timings.duration);
  if (response.status === 200) status200.add(1);
  if (response.status === 429) {
    status429.add(1);
    if (response.headers["Retry-After"]) retryAfter429.add(1);
  }
  if (response.status === 500) status500.add(1);
  if (response.status === 503) status503.add(1);
  sleep(0.01);
}

export function handleSummary(data) {
  return {
    stdout: `FORGE_K6_SUMMARY_JSON ${JSON.stringify(data)}\n`,
  };
}
