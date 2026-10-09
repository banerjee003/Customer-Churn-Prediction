-- ==============================================================================
-- Database & Table Schema: Telco Customer Churn Prediction
-- Database Engine: Microsoft SQL Server (T-SQL)
-- Database Name: customerChurnDB
-- Table Name: dbo.CustomerChurn
-- ==============================================================================

-- 1. Create Database (if not already existing)
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'customerChurnDB')
BEGIN
    CREATE DATABASE customerChurnDB;
END
GO

USE customerChurnDB;
GO

-- 2. Drop existing table if recreating
IF OBJECT_ID(N'dbo.CustomerChurn', N'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.CustomerChurn;
END
GO

-- 3. Create dbo.CustomerChurn Table
CREATE TABLE dbo.CustomerChurn (
    -- Customer Demographics & Identifiers
    customerID          VARCHAR(10)     NOT NULL,
    gender              VARCHAR(10)     NOT NULL,
    SeniorCitizen       BIT             NOT NULL,
    Partner             VARCHAR(3)      NOT NULL,
    Dependents          VARCHAR(3)      NOT NULL,

    -- Customer Tenure & Account Information
    tenure              INT             NOT NULL,

    -- Phone & Communication Services
    PhoneService        VARCHAR(3)      NOT NULL,
    MultipleLines       VARCHAR(20)     NOT NULL,

    -- Internet & Value-Added Services
    InternetService     VARCHAR(20)     NOT NULL,
    OnlineSecurity      VARCHAR(25)     NOT NULL,
    OnlineBackup        VARCHAR(25)     NOT NULL,
    DeviceProtection    VARCHAR(25)     NOT NULL,
    TechSupport         VARCHAR(25)     NOT NULL,
    StreamingTV         VARCHAR(25)     NOT NULL,
    StreamingMovies     VARCHAR(25)     NOT NULL,

    -- Contract & Billing Information
    Contract            VARCHAR(20)     NOT NULL,
    PaperlessBilling    VARCHAR(3)      NOT NULL,
    PaymentMethod       VARCHAR(35)     NOT NULL,
    MonthlyCharges      DECIMAL(10, 2)  NOT NULL,
    TotalCharges        DECIMAL(10, 2)  NULL,       -- NULL for new customers with tenure = 0

    -- Target Variable
    Churn               VARCHAR(3)      NOT NULL,

    -- Primary Key Constraint
    CONSTRAINT PK_CustomerChurn_customerID PRIMARY KEY CLUSTERED (customerID),

    -- Domain & Integrity Constraints
    CONSTRAINT CK_CustomerChurn_gender CHECK (gender IN ('Male', 'Female')),
    CONSTRAINT CK_CustomerChurn_Partner CHECK (Partner IN ('Yes', 'No')),
    CONSTRAINT CK_CustomerChurn_Dependents CHECK (Dependents IN ('Yes', 'No')),
    CONSTRAINT CK_CustomerChurn_tenure CHECK (tenure >= 0),
    CONSTRAINT CK_CustomerChurn_PhoneService CHECK (PhoneService IN ('Yes', 'No')),
    CONSTRAINT CK_CustomerChurn_Contract CHECK (Contract IN ('Month-to-month', 'One year', 'Two year')),
    CONSTRAINT CK_CustomerChurn_PaperlessBilling CHECK (PaperlessBilling IN ('Yes', 'No')),
    CONSTRAINT CK_CustomerChurn_Churn CHECK (Churn IN ('Yes', 'No'))
);
GO

-- ==============================================================================
-- 4. Performance Indexes
-- ==============================================================================

-- Index for target variable filtering and churn analysis
CREATE NONCLUSTERED INDEX IX_CustomerChurn_Churn
ON dbo.CustomerChurn (Churn);
GO

-- Index for contract and tenure segmentations
CREATE NONCLUSTERED INDEX IX_CustomerChurn_Contract_Tenure
ON dbo.CustomerChurn (Contract, tenure)
INCLUDE (MonthlyCharges, TotalCharges, Churn);
GO

-- Index for internet service subscriptions
CREATE NONCLUSTERED INDEX IX_CustomerChurn_InternetService
ON dbo.CustomerChurn (InternetService)
INCLUDE (Churn);
GO

-- ==============================================================================
-- 5. Data Ingestion Guide (Optional BULK INSERT Template)
-- ==============================================================================
-- If populating from dataset/Telco_Customer_Churn.csv directly via SQL Server:
--
-- BULK INSERT dbo.CustomerChurn
-- FROM 'YOUR-PATH\Telco_Customer_Churn.csv'
-- WITH (
--     FORMAT = 'CSV',
--     FIRSTROW = 2,
--     FIELDTERMINATOR = ',',
--     ROWTERMINATOR = '0x0a',
--     KEEPNULLS,
--     TABLOCK
-- );
-- GO
