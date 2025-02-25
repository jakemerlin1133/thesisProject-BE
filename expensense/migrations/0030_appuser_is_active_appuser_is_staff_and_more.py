# Generated by Django 5.1 on 2025-01-05 13:56

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "expensense",
            "0029_remove_appuser_is_active_remove_appuser_is_staff_and_more",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="appuser",
            name="is_active",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="appuser",
            name="is_staff",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="appuser",
            name="is_superuser",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="appuser",
            name="last_login",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="expense",
            name="uploaded_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
