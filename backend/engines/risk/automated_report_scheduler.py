#!/usr/bin/env python3
"""
Automated Risk Report Scheduler
Enterprise-grade automated report generation and delivery system for institutional clients

Features:
- Scheduled report generation (daily, weekly, monthly, custom)
- Email delivery with professional formatting
- Report caching and versioning
- Client-specific configurations
- Multi-format support (HTML, PDF-ready, JSON)
- Failure recovery and notification
"""

import asyncio
import logging
import smtplib
import os
import json
from datetime import datetime, timedelta, time as datetime_time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import aiofiles
import aiofiles.os
from pathlib import Path
import hashlib

from professional_risk_reporter import (
    ProfessionalRiskReporter,
    ReportConfiguration,
    ReportType,
    ReportFormat
)

logger = logging.getLogger(__name__)

class ScheduleFrequency(Enum):
    """Report generation schedule frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"

class DeliveryMethod(Enum):
    """Report delivery methods"""
    EMAIL = "email"
    S3_UPLOAD = "s3_upload"
    FTP_UPLOAD = "ftp_upload"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"

@dataclass
class ReportSchedule:
    """Report generation schedule configuration"""
    schedule_id: str
    portfolio_id: str
    client_name: str
    report_type: ReportType
    report_format: ReportFormat
    frequency: ScheduleFrequency
    delivery_method: DeliveryMethod
    
    # Schedule timing
    time_of_day: datetime_time = datetime_time(8, 0)  # 8:00 AM
    day_of_week: int = 0  # Monday
    day_of_month: int = 1  # First day of month
    custom_cron: Optional[str] = None
    
    # Delivery configuration
    email_recipients: List[str] = None
    email_subject_template: str = "Risk Analytics Report - {portfolio_id} - {date}"
    s3_bucket: Optional[str] = None
    s3_key_prefix: Optional[str] = None
    webhook_url: Optional[str] = None
    file_path_template: Optional[str] = None
    
    # Report configuration
    benchmark_symbol: str = "SPY"
    date_range_days: int = 252
    include_charts: bool = True
    include_attribution: bool = True
    
    # Metadata
    enabled: bool = True
    created_at: datetime = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.next_run is None:
            self.next_run = self._calculate_next_run()
    
    def _calculate_next_run(self) -> datetime:
        """Calculate next scheduled run time"""
        now = datetime.now()
        
        if self.frequency == ScheduleFrequency.DAILY:
            next_run = now.replace(
                hour=self.time_of_day.hour,
                minute=self.time_of_day.minute,
                second=0,
                microsecond=0
            )
            if next_run <= now:
                next_run += timedelta(days=1)
        
        elif self.frequency == ScheduleFrequency.WEEKLY:
            days_ahead = (self.day_of_week - now.weekday()) % 7
            if days_ahead == 0 and now.time() > self.time_of_day:
                days_ahead = 7
            next_run = now.replace(
                hour=self.time_of_day.hour,
                minute=self.time_of_day.minute,
                second=0,
                microsecond=0
            ) + timedelta(days=days_ahead)
        
        elif self.frequency == ScheduleFrequency.MONTHLY:
            if now.day < self.day_of_month or (
                now.day == self.day_of_month and now.time() <= self.time_of_day
            ):
                next_run = now.replace(
                    day=self.day_of_month,
                    hour=self.time_of_day.hour,
                    minute=self.time_of_day.minute,
                    second=0,
                    microsecond=0
                )
            else:
                # Next month
                if now.month == 12:
                    next_run = now.replace(
                        year=now.year + 1,
                        month=1,
                        day=self.day_of_month,
                        hour=self.time_of_day.hour,
                        minute=self.time_of_day.minute,
                        second=0,
                        microsecond=0
                    )
                else:
                    next_run = now.replace(
                        month=now.month + 1,
                        day=self.day_of_month,
                        hour=self.time_of_day.hour,
                        minute=self.time_of_day.minute,
                        second=0,
                        microsecond=0
                    )
        
        elif self.frequency == ScheduleFrequency.QUARTERLY:
            # Next quarter first day
            quarter_starts = [
                datetime(now.year, 1, 1),
                datetime(now.year, 4, 1),
                datetime(now.year, 7, 1),
                datetime(now.year, 10, 1),
                datetime(now.year + 1, 1, 1)
            ]
            
            next_run = None
            for quarter_start in quarter_starts:
                candidate = quarter_start.replace(
                    hour=self.time_of_day.hour,
                    minute=self.time_of_day.minute
                )
                if candidate > now:
                    next_run = candidate
                    break
        
        else:  # CUSTOM - would need cron parsing
            next_run = now + timedelta(days=1)  # Fallback
        
        return next_run

@dataclass
class ReportDelivery:
    """Report delivery tracking"""
    delivery_id: str
    schedule_id: str
    portfolio_id: str
    report_content: str
    report_format: ReportFormat
    delivery_method: DeliveryMethod
    
    # Delivery status
    status: str = "pending"  # pending, sent, failed
    delivery_timestamp: Optional[datetime] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    
    # File information
    file_size: int = 0
    file_hash: Optional[str] = None

class AutomatedReportScheduler:
    """
    Enterprise automated report scheduler
    Handles scheduled generation and delivery of professional risk reports
    """
    
    def __init__(self, professional_reporter: ProfessionalRiskReporter):
        """Initialize automated report scheduler"""
        self.professional_reporter = professional_reporter
        self.schedules: Dict[str, ReportSchedule] = {}
        self.deliveries: Dict[str, ReportDelivery] = {}
        self.is_running = False
        
        # Configuration
        self.reports_directory = Path("./report_cache")
        self.reports_directory.mkdir(exist_ok=True)
        
        self.config_file = Path("./scheduler_config.json")
        
        # Email configuration (would be loaded from environment)
        self.smtp_server = os.getenv("SMTP_SERVER", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_from_email = os.getenv("SMTP_FROM_EMAIL", "risk-analytics@nautilus.ai")
        
        # Statistics
        self.reports_generated = 0
        self.reports_delivered = 0
        self.delivery_failures = 0
        
        logger.info("Automated Report Scheduler initialized")
    
    async def start_scheduler(self):
        """Start the automated report scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        logger.info("Starting Automated Report Scheduler")
        
        # Load existing schedules
        await self._load_schedules()
        
        # Start scheduler loop
        asyncio.create_task(self._scheduler_loop())
        
        logger.info("Automated Report Scheduler started successfully")
    
    async def stop_scheduler(self):
        """Stop the automated report scheduler"""
        logger.info("Stopping Automated Report Scheduler")
        self.is_running = False
        
        # Save schedules
        await self._save_schedules()
        
        logger.info("Automated Report Scheduler stopped")
    
    async def add_schedule(self, schedule: ReportSchedule) -> str:
        """Add a new report schedule"""
        try:
            # Validate schedule
            if not schedule.portfolio_id:
                raise ValueError("Portfolio ID is required")
            
            if not schedule.client_name:
                raise ValueError("Client name is required")
            
            if schedule.delivery_method == DeliveryMethod.EMAIL and not schedule.email_recipients:
                raise ValueError("Email recipients required for email delivery")
            
            # Generate schedule ID if not provided
            if not schedule.schedule_id:
                schedule.schedule_id = self._generate_schedule_id(schedule)
            
            # Calculate next run
            schedule.next_run = schedule._calculate_next_run()
            
            # Store schedule
            self.schedules[schedule.schedule_id] = schedule
            
            # Save to persistence
            await self._save_schedules()
            
            logger.info(f"Added report schedule {schedule.schedule_id} for portfolio {schedule.portfolio_id}")
            return schedule.schedule_id
            
        except Exception as e:
            logger.error(f"Error adding schedule: {e}")
            raise
    
    async def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a report schedule"""
        try:
            if schedule_id in self.schedules:
                del self.schedules[schedule_id]
                await self._save_schedules()
                logger.info(f"Removed schedule {schedule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing schedule: {e}")
            raise
    
    async def update_schedule(self, schedule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing schedule"""
        try:
            if schedule_id not in self.schedules:
                return False
            
            schedule = self.schedules[schedule_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)
            
            # Recalculate next run
            schedule.next_run = schedule._calculate_next_run()
            
            await self._save_schedules()
            logger.info(f"Updated schedule {schedule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating schedule: {e}")
            raise
    
    async def get_schedule(self, schedule_id: str) -> Optional[ReportSchedule]:
        """Get a specific schedule"""
        return self.schedules.get(schedule_id)
    
    async def list_schedules(self, portfolio_id: Optional[str] = None) -> List[ReportSchedule]:
        """List all schedules, optionally filtered by portfolio"""
        schedules = list(self.schedules.values())
        
        if portfolio_id:
            schedules = [s for s in schedules if s.portfolio_id == portfolio_id]
        
        return schedules
    
    async def generate_immediate_report(self, schedule_id: str) -> str:
        """Generate and deliver report immediately"""
        try:
            if schedule_id not in self.schedules:
                raise ValueError(f"Schedule {schedule_id} not found")
            
            schedule = self.schedules[schedule_id]
            
            # Generate report
            report_content = await self._generate_scheduled_report(schedule)
            
            # Create delivery record
            delivery = ReportDelivery(
                delivery_id=self._generate_delivery_id(schedule),
                schedule_id=schedule_id,
                portfolio_id=schedule.portfolio_id,
                report_content=report_content,
                report_format=schedule.report_format,
                delivery_method=schedule.delivery_method,
                file_size=len(report_content) if isinstance(report_content, str) else len(str(report_content)),
                file_hash=self._calculate_hash(report_content)
            )
            
            # Deliver report
            await self._deliver_report(schedule, delivery)
            
            # Update statistics
            self.reports_generated += 1
            if delivery.status == "sent":
                self.reports_delivered += 1
            else:
                self.delivery_failures += 1
            
            logger.info(f"Immediate report generated and delivered for schedule {schedule_id}")
            return delivery.delivery_id
            
        except Exception as e:
            logger.error(f"Error generating immediate report: {e}")
            raise
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Starting scheduler loop")
        
        while self.is_running:
            try:
                await self._process_scheduled_reports()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Continue after error
        
        logger.info("Scheduler loop stopped")
    
    async def _process_scheduled_reports(self):
        """Process due scheduled reports"""
        now = datetime.now()
        
        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue
            
            if schedule.next_run and schedule.next_run <= now:
                try:
                    await self._execute_scheduled_report(schedule)
                    
                except Exception as e:
                    logger.error(f"Error executing scheduled report {schedule.schedule_id}: {e}")
                    schedule.failure_count += 1
                
                finally:
                    # Update next run time regardless of success/failure
                    schedule.next_run = schedule._calculate_next_run()
                    schedule.last_run = now
    
    async def _execute_scheduled_report(self, schedule: ReportSchedule):
        """Execute a single scheduled report"""
        try:
            logger.info(f"Executing scheduled report for {schedule.schedule_id}")
            
            # Generate report
            report_content = await self._generate_scheduled_report(schedule)
            
            # Create delivery record
            delivery = ReportDelivery(
                delivery_id=self._generate_delivery_id(schedule),
                schedule_id=schedule.schedule_id,
                portfolio_id=schedule.portfolio_id,
                report_content=report_content,
                report_format=schedule.report_format,
                delivery_method=schedule.delivery_method,
                file_size=len(report_content) if isinstance(report_content, str) else len(str(report_content)),
                file_hash=self._calculate_hash(report_content)
            )
            
            # Store delivery record
            self.deliveries[delivery.delivery_id] = delivery
            
            # Cache report
            await self._cache_report(schedule, delivery, report_content)
            
            # Deliver report
            await self._deliver_report(schedule, delivery)
            
            # Update statistics
            schedule.run_count += 1
            self.reports_generated += 1
            
            if delivery.status == "sent":
                self.reports_delivered += 1
                schedule.failure_count = 0  # Reset failure count on success
            else:
                self.delivery_failures += 1
                schedule.failure_count += 1
            
            logger.info(f"Scheduled report executed successfully for {schedule.schedule_id}")
            
        except Exception as e:
            logger.error(f"Error executing scheduled report: {e}")
            schedule.failure_count += 1
            raise
    
    async def _generate_scheduled_report(self, schedule: ReportSchedule) -> str:
        """Generate report for a schedule"""
        try:
            # Create report configuration
            config = ReportConfiguration(
                report_type=schedule.report_type,
                format=schedule.report_format,
                date_range_days=schedule.date_range_days,
                benchmark_symbol=schedule.benchmark_symbol,
                include_charts=schedule.include_charts,
                include_attribution=schedule.include_attribution,
                custom_branding={
                    "client_name": schedule.client_name,
                    "report_title": f"{schedule.client_name} - Risk Analytics Report",
                    "footer_text": f"Prepared for {schedule.client_name}"
                }
            )
            
            # Generate report
            report_content = await self.professional_reporter.generate_professional_report(
                portfolio_id=schedule.portfolio_id,
                config=config
            )
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating scheduled report: {e}")
            raise
    
    async def _deliver_report(self, schedule: ReportSchedule, delivery: ReportDelivery):
        """Deliver report according to schedule configuration"""
        try:
            if schedule.delivery_method == DeliveryMethod.EMAIL:
                await self._deliver_email(schedule, delivery)
            
            elif schedule.delivery_method == DeliveryMethod.FILE_SYSTEM:
                await self._deliver_file_system(schedule, delivery)
            
            elif schedule.delivery_method == DeliveryMethod.WEBHOOK:
                await self._deliver_webhook(schedule, delivery)
            
            # Add more delivery methods as needed
            
        except Exception as e:
            delivery.status = "failed"
            delivery.failure_reason = str(e)
            logger.error(f"Error delivering report: {e}")
            raise
    
    async def _deliver_email(self, schedule: ReportSchedule, delivery: ReportDelivery):
        """Deliver report via email"""
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.smtp_from_email
            msg['To'] = ', '.join(schedule.email_recipients)
            msg['Subject'] = schedule.email_subject_template.format(
                portfolio_id=schedule.portfolio_id,
                client_name=schedule.client_name,
                date=datetime.now().strftime('%Y-%m-%d')
            )
            
            # Email body
            body = f"""
            Dear {schedule.client_name},
            
            Please find attached your automated risk analytics report for portfolio {schedule.portfolio_id}.
            
            Report Details:
            - Portfolio: {schedule.portfolio_id}
            - Report Type: {schedule.report_type.value}
            - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - Period: {schedule.date_range_days} days
            
            This report contains comprehensive risk analytics and performance metrics.
            
            Best regards,
            Nautilus Risk Analytics Team
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Attach report
            if schedule.report_format == ReportFormat.HTML:
                attachment = MimeText(delivery.report_content, 'html')
                attachment.add_header('Content-Disposition', 
                                    f'attachment; filename="risk_report_{schedule.portfolio_id}_{datetime.now().strftime("%Y%m%d")}.html"')
            else:
                attachment = MimeBase('application', 'octet-stream')
                attachment.set_payload(delivery.report_content)
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', 
                                    f'attachment; filename="risk_report_{schedule.portfolio_id}_{datetime.now().strftime("%Y%m%d")}.json"')
            
            msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            delivery.status = "sent"
            delivery.delivery_timestamp = datetime.now()
            
            logger.info(f"Email delivered successfully to {schedule.email_recipients}")
            
        except Exception as e:
            delivery.status = "failed"
            delivery.failure_reason = f"Email delivery failed: {str(e)}"
            logger.error(f"Email delivery failed: {e}")
            raise
    
    async def _deliver_file_system(self, schedule: ReportSchedule, delivery: ReportDelivery):
        """Deliver report to file system"""
        try:
            if not schedule.file_path_template:
                raise ValueError("File path template required for file system delivery")
            
            # Generate file path
            file_path = schedule.file_path_template.format(
                portfolio_id=schedule.portfolio_id,
                client_name=schedule.client_name.replace(' ', '_'),
                date=datetime.now().strftime('%Y%m%d'),
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write report content
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                if isinstance(delivery.report_content, str):
                    await f.write(delivery.report_content)
                else:
                    await f.write(json.dumps(delivery.report_content, indent=2))
            
            delivery.status = "sent"
            delivery.delivery_timestamp = datetime.now()
            
            logger.info(f"Report delivered to file system: {file_path}")
            
        except Exception as e:
            delivery.status = "failed"
            delivery.failure_reason = f"File system delivery failed: {str(e)}"
            logger.error(f"File system delivery failed: {e}")
            raise
    
    async def _deliver_webhook(self, schedule: ReportSchedule, delivery: ReportDelivery):
        """Deliver report via webhook"""
        try:
            import aiohttp
            
            if not schedule.webhook_url:
                raise ValueError("Webhook URL required for webhook delivery")
            
            # Prepare payload
            payload = {
                "portfolio_id": schedule.portfolio_id,
                "client_name": schedule.client_name,
                "report_type": schedule.report_type.value,
                "report_format": schedule.report_format.value,
                "generated_at": datetime.now().isoformat(),
                "report_content": delivery.report_content
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    schedule.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status < 300:
                        delivery.status = "sent"
                        delivery.delivery_timestamp = datetime.now()
                        logger.info(f"Webhook delivered successfully to {schedule.webhook_url}")
                    else:
                        raise Exception(f"Webhook returned status {response.status}")
            
        except Exception as e:
            delivery.status = "failed"
            delivery.failure_reason = f"Webhook delivery failed: {str(e)}"
            logger.error(f"Webhook delivery failed: {e}")
            raise
    
    async def _cache_report(self, schedule: ReportSchedule, delivery: ReportDelivery, content: str):
        """Cache generated report"""
        try:
            cache_file = self.reports_directory / f"{delivery.delivery_id}.cache"
            
            cache_data = {
                "delivery_id": delivery.delivery_id,
                "schedule_id": schedule.schedule_id,
                "portfolio_id": schedule.portfolio_id,
                "generated_at": datetime.now().isoformat(),
                "report_format": schedule.report_format.value,
                "file_size": delivery.file_size,
                "file_hash": delivery.file_hash,
                "content": content
            }
            
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(cache_data, indent=2))
            
            logger.debug(f"Report cached: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache report: {e}")
    
    async def _load_schedules(self):
        """Load schedules from persistence"""
        try:
            if self.config_file.exists():
                async with aiofiles.open(self.config_file, 'r') as f:
                    data = json.loads(await f.read())
                    
                    for schedule_data in data.get("schedules", []):
                        # Convert string dates back to datetime objects
                        if "created_at" in schedule_data and schedule_data["created_at"]:
                            schedule_data["created_at"] = datetime.fromisoformat(schedule_data["created_at"])
                        if "last_run" in schedule_data and schedule_data["last_run"]:
                            schedule_data["last_run"] = datetime.fromisoformat(schedule_data["last_run"])
                        if "next_run" in schedule_data and schedule_data["next_run"]:
                            schedule_data["next_run"] = datetime.fromisoformat(schedule_data["next_run"])
                        
                        # Convert time_of_day
                        if "time_of_day" in schedule_data and schedule_data["time_of_day"]:
                            time_parts = schedule_data["time_of_day"].split(':')
                            schedule_data["time_of_day"] = datetime_time(
                                int(time_parts[0]), 
                                int(time_parts[1])
                            )
                        
                        # Convert enums
                        schedule_data["report_type"] = ReportType(schedule_data["report_type"])
                        schedule_data["report_format"] = ReportFormat(schedule_data["report_format"])
                        schedule_data["frequency"] = ScheduleFrequency(schedule_data["frequency"])
                        schedule_data["delivery_method"] = DeliveryMethod(schedule_data["delivery_method"])
                        
                        schedule = ReportSchedule(**schedule_data)
                        self.schedules[schedule.schedule_id] = schedule
                
                logger.info(f"Loaded {len(self.schedules)} schedules from persistence")
                
        except Exception as e:
            logger.warning(f"Failed to load schedules: {e}")
    
    async def _save_schedules(self):
        """Save schedules to persistence"""
        try:
            schedules_data = []
            
            for schedule in self.schedules.values():
                schedule_dict = asdict(schedule)
                
                # Convert datetime objects to strings
                if schedule_dict["created_at"]:
                    schedule_dict["created_at"] = schedule_dict["created_at"].isoformat()
                if schedule_dict["last_run"]:
                    schedule_dict["last_run"] = schedule_dict["last_run"].isoformat()
                if schedule_dict["next_run"]:
                    schedule_dict["next_run"] = schedule_dict["next_run"].isoformat()
                
                # Convert time_of_day
                if schedule_dict["time_of_day"]:
                    schedule_dict["time_of_day"] = schedule_dict["time_of_day"].strftime("%H:%M")
                
                # Convert enums to values
                schedule_dict["report_type"] = schedule_dict["report_type"].value
                schedule_dict["report_format"] = schedule_dict["report_format"].value
                schedule_dict["frequency"] = schedule_dict["frequency"].value
                schedule_dict["delivery_method"] = schedule_dict["delivery_method"].value
                
                schedules_data.append(schedule_dict)
            
            data = {"schedules": schedules_data}
            
            async with aiofiles.open(self.config_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            logger.debug(f"Saved {len(self.schedules)} schedules to persistence")
            
        except Exception as e:
            logger.error(f"Failed to save schedules: {e}")
    
    def _generate_schedule_id(self, schedule: ReportSchedule) -> str:
        """Generate unique schedule ID"""
        unique_str = f"{schedule.portfolio_id}_{schedule.client_name}_{schedule.report_type.value}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def _generate_delivery_id(self, schedule: ReportSchedule) -> str:
        """Generate unique delivery ID"""
        unique_str = f"{schedule.schedule_id}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate content hash"""
        if isinstance(content, str):
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        else:
            return hashlib.sha256(str(content).encode()).hexdigest()[:16]
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics"""
        return {
            "is_running": self.is_running,
            "total_schedules": len(self.schedules),
            "enabled_schedules": len([s for s in self.schedules.values() if s.enabled]),
            "reports_generated": self.reports_generated,
            "reports_delivered": self.reports_delivered,
            "delivery_failures": self.delivery_failures,
            "success_rate": (self.reports_delivered / max(self.reports_generated, 1)) * 100,
            "next_scheduled_runs": [
                {
                    "schedule_id": s.schedule_id,
                    "portfolio_id": s.portfolio_id,
                    "next_run": s.next_run.isoformat() if s.next_run else None
                }
                for s in sorted(self.schedules.values(), key=lambda x: x.next_run or datetime.max)[:5]
            ]
        }

# Factory function
async def create_automated_scheduler(professional_reporter: ProfessionalRiskReporter) -> AutomatedReportScheduler:
    """Create and initialize automated report scheduler"""
    scheduler = AutomatedReportScheduler(professional_reporter)
    await scheduler.start_scheduler()
    return scheduler