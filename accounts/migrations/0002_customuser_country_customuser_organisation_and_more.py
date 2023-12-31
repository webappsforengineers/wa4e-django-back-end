# Generated by Django 4.2.6 on 2023-11-08 15:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='customuser',
            name='country',
            field=models.CharField(default='na', max_length=255),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='customuser',
            name='organisation',
            field=models.CharField(default='na', max_length=255),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='customuser',
            name='first_name',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='customuser',
            name='last_name',
            field=models.CharField(max_length=255),
        ),
    ]
