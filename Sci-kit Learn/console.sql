USE datasets;

SELECT *
FROM datasets.loantrain
WHERE Gender = "Male";


SELECT *
FROM datasets.loantrain
WHERE Gender = "Male" or Married = "Yes";

DESCRIBE datasets.loantrain;

#To make a frequency table of a column
SELECT DAY_OF_WEEK, count(*)
FROM boston
GROUP BY DAY_OF_WEEK;



