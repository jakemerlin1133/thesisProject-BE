# Generated by Django 5.1 on 2024-12-25 08:14

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("expensense", "0021_alter_expense_uploaded_at"),
    ]

    operations = [
        migrations.AlterField(
            model_name="expense",
            name="uploaded_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]