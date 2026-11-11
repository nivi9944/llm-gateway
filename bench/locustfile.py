"""Locust user for bench/load.py.

Every request has a unique question, so each one is a cache MISS and takes the full path
(auth, rate limit, exact + semantic lookup, provider call, cache store), the worst case for
gateway overhead.

The same file runs against the gateway and directly against the mock (the mock ignores the
auth header), which is how load.py measures the latency the gateway adds.
"""
import os
import random
import uuid

from locust import HttpUser, constant, events, task

KEY = os.getenv("GATEWAY_KEY", "bench-key")
RAW_OUT = os.getenv("LOCUST_RAW_OUT")  # if set, every response time (ms) is saved here
_times: list[float] = []

# Random word mixes, so no two prompts mean the same thing. Otherwise the semantic cache
# would answer some of them and the full MISS path would not be measured.
WORDS = """apple river engine castle violin planet garden copper ladder winter pepper rocket
harbor blanket candle desert falcon glacier hammer island jungle kettle lantern marble
needle orchard pillow quartz saddle tunnel umbrella valley wagon yogurt zebra anchor
bridge cactus dolphin feather ginger helmet igloo jacket kitten lemon magnet nectar
oyster parrot quiver radar salmon tractor unicorn velvet walnut xylophone yacht zipper
algebra ballet cricket diamond eclipse fossil gravity horizon insect justice karma
leopard mustard nebula opera pyramid quantum rhythm satellite tornado uranium vaccine
whistle cobalt tundra meadow canyon compass blizzard biscuit bamboo asteroid avocado
banjo barometer beetle bicycle bonsai buffalo cabbage camera canoe carpet cathedral
cello cheetah chimney cinnamon citadel clover coconut comet coral cotton crayon
crystal cymbal daisy dragon drum eagle elbow emerald falafel fern fiddle flamingo""".split()


def random_prompt() -> str:
    return " ".join(random.sample(WORDS, 8)) + " " + uuid.uuid4().hex[:8]


@events.request.add_listener
def _record(response_time, exception, **_):
    # Locust's own CSV rounds times above 100 ms, so the exact values are kept as well.
    if exception is None:
        _times.append(response_time)


@events.spawning_complete.add_listener
def _reset(user_count, **_):
    # Measure only once all users are running, so the ramp-up is excluded as with --reset-stats.
    _times.clear()


@events.quitting.add_listener
def _dump(environment, **_):
    if RAW_OUT:
        with open(RAW_OUT, "w") as f:
            f.write("\n".join(f"{t:.3f}" for t in _times))


class ChatUser(HttpUser):
    wait_time = constant(0)  # closed loop: each user sends the next request as soon as one returns

    @task
    def chat(self):
        body = {
            "model": "auto",
            "temperature": 0,
            "messages": [{"role": "user", "content": random_prompt()}],
        }
        self.client.post("/v1/chat/completions", json=body, name="chat",
                         headers={"Authorization": f"Bearer {KEY}"})
