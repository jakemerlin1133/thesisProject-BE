# Generated by Django 5.1 on 2024-12-24 07:39

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("expensense", "0016_alter_expense_matched_store_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="expense",
            name="matched_store",
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="expense",
            name="matched_store_category",
            field=models.JSONField(blank=True, null=True),
        ),
    ]