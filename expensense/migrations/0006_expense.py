# Generated by Django 5.1 on 2024-12-23 04:01

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("expensense", "0005_store"),
    ]

    operations = [
        migrations.CreateModel(
            name="Expense",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("total_total", models.DecimalField(decimal_places=2, max_digits=10)),
                (
                    "catergory_id",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="expensense.category",
                    ),
                ),
                (
                    "store_id",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="expensense.store",
                    ),
                ),
            ],
        ),
    ]