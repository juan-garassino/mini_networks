# €25 cap with 40/80/100% alerts. Off by default (needs billing-account admin;
# account-level alerts already exist on garassino-ml). Enable with manage_budget.
resource "google_billing_budget" "cap" {
  count           = var.manage_budget ? 1 : 0
  billing_account = var.billing_account
  display_name    = "mini-networks cap"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "EUR"
      units         = var.budget_amount_eur
    }
  }

  threshold_rules {
    threshold_percent = 0.4
  }
  threshold_rules {
    threshold_percent = 0.8
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}
