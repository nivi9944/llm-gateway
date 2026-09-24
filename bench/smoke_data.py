"""A tiny hand-written set of question pairs, used ONLY by `--smoke` runs.

Smoke runs check that the benchmark code works end to end without downloading anything.
Their numbers mean nothing and are never written to results/.
label 1 = same meaning (duplicate), label 0 = different meaning.
"""

PAIRS = [
    ("How do I learn Python quickly?", "What is the fastest way to learn Python?", 1),
    ("How can I lose weight fast?", "What are quick ways to lose weight?", 1),
    ("What is the best way to learn guitar?", "How should a beginner learn guitar?", 1),
    ("How do I improve my English speaking?", "How can I get better at speaking English?", 1),
    ("Why is the sky blue?", "What makes the sky look blue?", 1),
    ("How do I start investing in stocks?", "How can a beginner start investing in the stock market?", 1),
    ("What is machine learning?", "Can someone explain what machine learning is?", 1),
    ("How can I become a good programmer?", "What should I do to become a great programmer?", 1),
    ("How do I make money online?", "What are ways to earn money on the internet?", 1),
    ("Which is the best laptop for students?", "What laptop should a student buy?", 1),
    ("How do I prepare for a job interview?", "What is the best way to prepare for an interview?", 1),
    ("How do black holes form?", "What causes a black hole to form?", 1),
    ("How do I learn Java?", "How do I learn JavaScript?", 0),
    ("What is the capital of Australia?", "What is the largest city in Australia?", 0),
    ("How do I cook rice?", "How do I cook pasta?", 0),
    ("Is it safe to travel to Brazil?", "Is it expensive to travel to Brazil?", 0),
    ("What are the benefits of running?", "What are the risks of running?", 0),
    ("How do I delete my Facebook account?", "How do I create a Facebook account?", 0),
    ("Who will win the next football world cup?", "Who won the last football world cup?", 0),
    ("How can I stop overthinking?", "How can I stop procrastinating?", 0),
    ("What is the best phone under 20000?", "What is the best laptop under 20000?", 0),
    ("How do I get a US visa?", "How long does a US visa take?", 0),
    ("What is quantum computing?", "What is cloud computing?", 0),
    ("Why do cats purr?", "Why do dogs bark?", 0),
]
