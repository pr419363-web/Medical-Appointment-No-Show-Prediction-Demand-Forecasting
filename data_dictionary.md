# Raw Data Dictionary

Schema for `data/raw/medical_appointments.csv`. Names below preserve the source CSV spelling. Binary fields use `0`/`1` unless otherwise noted. Blank values may be present; categorical vocabularies should be read from the source data rather than assumed.

| Column | Type | Description |
|---|---|---|
| `specialty` | string | Appointment specialty/service. |
| `appointment_time` | integer | Appointment hour (24-hour clock). |
| `gender` | category | Patient gender code as provided by the source. |
| `no_show` | category | Attendance outcome: `yes` indicates a no-show; `no` indicates attendance. |
| `disability` | category | Disability classification. |
| `place` | string | Raw location label; 26,289 distinct non-null strings occur in the supplied CSV, with mixed formats. A canonical city mapping is not provided. |
| `appointment_shift` | category | Appointment shift. |
| `age` | numeric | Patient age in years. |
| `under_12_years_old` | boolean/integer | Indicator that age is under 12. |
| `over_60_years_old` | boolean/integer | Indicator that age is over 60. |
| `patient_needs_companion` | boolean/integer | Indicator that a companion is needed. |
| `average_temp_day` | numeric | Average temperature for the appointment day (°C). |
| `average_rain_day` | numeric | Average rainfall for the appointment day (source units). |
| `max_temp_day` | numeric | Maximum temperature for the appointment day (°C). |
| `max_rain_day` | numeric | Maximum rainfall for the appointment day (source units). |
| `rainy_day_before` | boolean/integer | Indicator that the previous day was rainy. |
| `storm_day_before` | boolean/integer | Indicator that the previous day had a storm. |
| `rain_intensity` | category | Rain-intensity category. |
| `heat_intensity` | category | Heat-intensity category. |
| `appointment_date_continuous` | date/datetime | Appointment date used for temporal grouping. |
| `Hipertension` | boolean/integer | Hypertension indicator; source spelling is retained. |
| `Diabetes` | boolean/integer | Diabetes indicator. |
| `Alcoholism` | boolean/integer | Alcoholism indicator. |
| `Handcap` | boolean/integer | Physical-handicap indicator; source spelling is retained. |
| `Scholarship` | boolean/integer | Scholarship/program enrollment indicator. |
| `SMS_received` | boolean/integer | Indicator that an SMS reminder was received. |

The supplied file contains both recognizable Brazilian municipality labels and synthetic-looking place names. Do not interpret each distinct string as a real city without cleaning and validating the source. Exact value encodings and null counts can vary by dataset version. This project currently reads CSV files directly; a database schema is not defined.