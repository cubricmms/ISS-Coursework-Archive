# Airbnb-bigdata-Singapore
### Team project based on the data provided by the Inside Airbnb

#### Columns used in this project:
* amenities
* bathrooms
* bed_type
* bedrooms
* beds
* cancellation_policy
* id
* latitude
* longitude
* neighbourhood_group_cleansed
* price
* property_type

* room_type

#### Columns decided drop after data exploration
* weekly_price
* monthly_price
* cleaning_fee (removed, too much missing)
* square_feet (removed, too much missing)
* review_scores_accuracy (removed, too much missing)
* review_scores_checkin (removed, too much missing)
* review_scores_cleanliness (removed, too much missing)
* review_scores_communication (removed, too much missing)
* review_scores_location (removed, too much missing)
* review_scores_rating (removed, too much missing)
* reviews_per_month (removed, too much missing)

#### Issues
`review_scores_accuracy` `review_scores_checkin` `review_scores_cleanliness` `review_scores_communication` `review_scores_location` `review_scores_rating` `reviews_per_month` may need to be reevaluated and added back if modeling fails
