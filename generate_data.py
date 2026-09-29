"""Builds a synthetic labelled email dataset (High / Medium / Low).
Includes deliberate noise: cross-class phrases, promo emails that SHOUT 'urgent',
and 7% label noise, so the task is not trivially separable.
Swap in a real corpus (Enron, etc.) by producing the same CSV columns."""
import random, pandas as pd
random.seed(42)
CORE = {
 "High": ["production server is down, need a fix ASAP", "deadline is today, submit the report by 5 PM",
          "client escalation: contract at risk, call me immediately", "payment failed, action required today",
          "interview scheduled tomorrow, confirm now", "exam rescheduled to tomorrow morning, please confirm",
          "security breach detected, respond immediately", "overdue invoice, final notice before suspension",
          "critical bug reported by customer, needs a patch urgently", "board meeting moved to tonight, join now"],
 "Medium": ["please review the meeting notes by Friday", "can you share the project update this week",
            "reminder: team sync on Thursday", "feedback requested on the draft proposal",
            "please approve my leave request when free", "sharing the agenda for next week's workshop",
            "could you check the attached spreadsheet", "following up on the earlier question about the budget",
            "assignment guidelines updated, take a look", "new task assigned to you in the tracker"],
 "Low": ["weekly newsletter: top 10 tech stories", "50% off sale this weekend only", "your monthly statement is available",
         "someone started following you", "cafeteria lunch menu for next week", "webinar invitation: join our upcoming event",
         "you may like these recommended products", "thanks for subscribing to our updates",
         "photos from the office party", "unsubscribe or manage your email preferences"],
}
OPEN = ["Hi,", "Hello team,", "Dear all,", "Hey,", "Good morning,", ""]
CLOSE = ["Thanks", "Regards", "Best", "Cheers", "Thank you", ""]
HARD_LOW = ["URGENT!!! limited-time offer, act now", "Last chance! sale ends today", "Action required: claim your free coupon now"]
HARD_HIGH = ["quick note, when you get a moment, this is time sensitive", "small thing but it needs sorting out before the deadline"]
SENDERS = {"High": ["ceo.office@corp.com", "client.rep@acme.com", "boss@corp.com", "hr.dept@corp.com", "bank.alerts@bank.com", "colleague@corp.com"],
           "Medium": ["colleague@corp.com", "manager@corp.com", "prof@university.edu", "hr.dept@corp.com", "team@corp.com"],
           "Low": ["news@media.com", "promo@shop.com", "no-reply@social.com", "offers@deals.com", "team@corp.com"]}
def make(label):
    r = random.random()
    src = label
    if r < 0.12: src = random.choice([l for l in CORE if l != label])   # cross-class noise
    core = random.choice(CORE[src])
    if label == "Low" and random.random() < 0.15: core = random.choice(HARD_LOW)
    if label == "High" and random.random() < 0.12: core = random.choice(HARD_HIGH)
    if random.random() < 0.3: core = core.upper() if random.random() < .3 else core.capitalize() + ("!" if random.random() < .3 else "")
    body = f"{random.choice(OPEN)} {core}. {random.choice(CORE[random.choice(list(CORE))]) if random.random()<.25 else ''} {random.choice(CLOSE)}".strip()
    subject = random.choice(CORE[src]).split(",")[0][:50]
    sender = random.choice(SENDERS[label] if random.random() < .8 else sum(SENDERS.values(), []))
    y = label if random.random() > 0.07 else random.choice(list(CORE))    # label noise
    return {"subject": subject, "body": body, "sender": sender, "label": y}
rows = [make(l) for l in CORE for _ in range(700)]
df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv("data/emails.csv", index=False)
print(df.label.value_counts()); print(df.head())
