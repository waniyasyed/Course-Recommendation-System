"""
Smart Course Recommendation System for University Students
Team: Waniya Syed (22k-4516) & Hafsa Atiqi (22k-4584)

Technique: Hybrid Recommendation (Collaborative Filtering + Content-Based)
- Collaborative Filtering: Item-based cosine similarity on student-course rating matrix
- Content-Based Filtering: TF-IDF + cosine similarity on course attributes
- Both are fully explainable (no black box)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. REAL DATASET (Student_Performance.csv)
# ─────────────────────────────────────────────
import os

CSV_PATH = "Student_Performance.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find '{CSV_PATH}'.\n"
        "Make sure Student_Performance.csv is in the same folder as this script."
    )

raw = pd.read_csv(CSV_PATH)

# Map real dataset columns to system features:
#   overall_score (0-100)  → GPA on 4.0 scale
#   study_method           → study_time proxy (mapped: online=8, group=6, self=4, tutoring=3, textbook=2)
#   attendance_percentage  → attendance (already 0-100)
#   math_score             → math_score (already 0-100)
#   english_score          → prog_score proxy (logical/verbal reasoning)
#   science_score          → net_score proxy (analytical thinking)

study_method_map = {
    "online": 8, "group": 6, "self-study": 4,
    "tutoring": 3, "textbook": 2
}

students_df = pd.DataFrame({
    "student_id": [f"S{str(i).zfill(3)}" for i in range(1, len(raw) + 1)],
    "gpa":        (raw["overall_score"] / 100 * 4).clip(1.0, 4.0).round(2),
    "study_time": raw["study_method"].str.lower().map(study_method_map).fillna(4.0),
    "attendance": raw["attendance_percentage"].clip(0, 100).round(1),
    "math_score": raw["math_score"].clip(0, 100).round(1),
    "prog_score": raw["english_score"].clip(0, 100).round(1),   # verbal/logical proxy
    "net_score":  raw["science_score"].clip(0, 100).round(1),   # analytical proxy
})

print(f"Loaded {len(students_df)} students from {CSV_PATH}")

# ─────────────────────────────────────────────
# 2. COURSE DATASET
# ─────────────────────────────────────────────
courses = [
    {"course_id": "C01", "name": "Calculus I",              "domain": "Math",       "difficulty": 2, "prereq_gpa": 1.5, "math_req": 40, "prog_req":  0, "net_req":  0},
    {"course_id": "C02", "name": "Calculus II",             "domain": "Math",       "difficulty": 3, "prereq_gpa": 2.0, "math_req": 60, "prog_req":  0, "net_req":  0},
    {"course_id": "C03", "name": "Linear Algebra",          "domain": "Math",       "difficulty": 3, "prereq_gpa": 2.0, "math_req": 55, "prog_req":  0, "net_req":  0},
    {"course_id": "C04", "name": "Intro to Programming",    "domain": "CS",         "difficulty": 1, "prereq_gpa": 1.0, "math_req": 20, "prog_req":  0, "net_req":  0},
    {"course_id": "C05", "name": "Data Structures",         "domain": "CS",         "difficulty": 3, "prereq_gpa": 2.0, "math_req": 40, "prog_req": 50, "net_req":  0},
    {"course_id": "C06", "name": "Algorithms",              "domain": "CS",         "difficulty": 4, "prereq_gpa": 2.5, "math_req": 65, "prog_req": 65, "net_req":  0},
    {"course_id": "C07", "name": "Machine Learning",        "domain": "AI",         "difficulty": 4, "prereq_gpa": 2.5, "math_req": 70, "prog_req": 70, "net_req":  0},
    {"course_id": "C08", "name": "Deep Learning",           "domain": "AI",         "difficulty": 5, "prereq_gpa": 3.0, "math_req": 80, "prog_req": 75, "net_req":  0},
    {"course_id": "C09", "name": "Computer Networks",       "domain": "Networks",   "difficulty": 3, "prereq_gpa": 2.0, "math_req": 30, "prog_req": 40, "net_req":  0},
    {"course_id": "C10", "name": "Network Security",        "domain": "Networks",   "difficulty": 4, "prereq_gpa": 2.5, "math_req": 40, "prog_req": 50, "net_req": 60},
    {"course_id": "C11", "name": "Database Systems",        "domain": "CS",         "difficulty": 3, "prereq_gpa": 2.0, "math_req": 35, "prog_req": 55, "net_req":  0},
    {"course_id": "C12", "name": "Operating Systems",       "domain": "CS",         "difficulty": 4, "prereq_gpa": 2.5, "math_req": 45, "prog_req": 60, "net_req":  0},
    {"course_id": "C13", "name": "Software Engineering",    "domain": "SE",         "difficulty": 3, "prereq_gpa": 2.0, "math_req": 25, "prog_req": 55, "net_req":  0},
    {"course_id": "C14", "name": "Computer Vision",         "domain": "AI",         "difficulty": 4, "prereq_gpa": 2.8, "math_req": 70, "prog_req": 65, "net_req":  0},
    {"course_id": "C15", "name": "Probability & Statistics","domain": "Math",       "difficulty": 3, "prereq_gpa": 2.0, "math_req": 55, "prog_req":  0, "net_req":  0},
]
courses_df = pd.DataFrame(courses)

# ─────────────────────────────────────────────
# 3. IMPLICIT RATING FUNCTION (Explainable)
# ─────────────────────────────────────────────
def compute_implicit_rating(student: pd.Series, course: pd.Series) -> float:
    """
    Explicit scoring formula — fully transparent, no black box.

    Score = w1*GPA_component + w2*subject_match + w3*effort_component - penalty
    Range: 0 – 5
    """
    w1, w2, w3 = 0.4, 0.4, 0.2

    # GPA component: how well does GPA exceed the prereq?
    gpa_margin = student["gpa"] - course["prereq_gpa"]
    gpa_comp = np.clip(gpa_margin / (4.0 - course["prereq_gpa"] + 1e-6), 0, 1)

    # Subject match: weighted average of relevant scores vs requirements
    reqs = {
        "math_score": course["math_req"],
        "prog_score": course["prog_req"],
        "net_score":  course["net_req"],
    }
    total_req = sum(reqs.values()) + 1e-6
    subject_match = sum(
        (student[col] / 100.0) * (req / total_req)
        for col, req in reqs.items()
    )
    subject_match = np.clip(subject_match, 0, 1)

    # Effort component: study time & attendance proxy
    effort = np.clip((student["study_time"] / 10.0) * 0.5 +
                     (student["attendance"] / 100.0) * 0.5, 0, 1)

    # Difficulty penalty for struggling students
    difficulty_penalty = 0.0
    if course["difficulty"] >= 4 and student["gpa"] < 2.5:
        difficulty_penalty = 0.5

    raw = w1 * gpa_comp + w2 * subject_match + w3 * effort - difficulty_penalty
    return round(float(np.clip(raw * 5, 0.5, 5.0)), 2)

# Build rating matrix
print("Building rating matrix...")
ratings = {}
for _, student in students_df.iterrows():
    ratings[student["student_id"]] = {
        row["course_id"]: compute_implicit_rating(student, row)
        for _, row in courses_df.iterrows()
    }

rating_matrix = pd.DataFrame(ratings).T  # shape: students x courses
rating_matrix.index.name = "student_id"

# ─────────────────────────────────────────────
# 4. COLLABORATIVE FILTERING (Item-Based)
# ─────────────────────────────────────────────
def collaborative_filtering_scores(student_id: str, rating_matrix: pd.DataFrame) -> pd.Series:
    """
    Item-based CF: for each course not yet 'taken', predict rating using
    weighted average of similar courses the student has rated highly.
    Fully explainable — similarity = cosine similarity between course rating vectors.
    """
    course_sim = pd.DataFrame(
        cosine_similarity(rating_matrix.T),
        index=rating_matrix.columns,
        columns=rating_matrix.columns,
    )
    student_ratings = rating_matrix.loc[student_id]
    predicted = {}
    for course in rating_matrix.columns:
        sims = course_sim[course].drop(course)
        rated = student_ratings.drop(course)
        # Weighted average of similar course ratings
        denom = sims.abs().sum()
        if denom < 1e-6:
            predicted[course] = student_ratings.mean()
        else:
            predicted[course] = (sims * rated).sum() / denom
    return pd.Series(predicted)


# ─────────────────────────────────────────────
# 5. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────
def content_based_scores(student: pd.Series, courses_df: pd.DataFrame) -> pd.Series:
    """
    Content-based: build a student preference vector from their academic profile,
    compute cosine similarity with each course's feature vector.
    """
    # Course feature matrix (normalised)
    scaler = MinMaxScaler()
    course_features = scaler.fit_transform(
        courses_df[["difficulty", "math_req", "prog_req", "net_req"]]
    )

    # Student preference vector
    student_vec = np.array([
        student["gpa"] / 4.0,
        student["math_score"] / 100.0,
        student["prog_score"] / 100.0,
        student["net_score"] / 100.0,
    ]).reshape(1, -1)

    sims = cosine_similarity(student_vec, course_features).flatten()
    return pd.Series(dict(zip(courses_df["course_id"], sims)))


# ─────────────────────────────────────────────
# 6. HYBRID RECOMMENDER
# ─────────────────────────────────────────────
def hybrid_recommend(student_id: str, top_n: int = 5, cf_weight: float = 0.6, cb_weight: float = 0.4):
    """
    Combine CF and CB scores using weighted sum.
    Returns top-N recommendations with score breakdown.
    """
    student = students_df.set_index("student_id").loc[student_id]

    cf_scores = collaborative_filtering_scores(student_id, rating_matrix)
    cb_scores  = content_based_scores(student, courses_df)

    # Normalise both to [0, 1]
    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-6)
    cb_norm = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min() + 1e-6)

    hybrid = cf_weight * cf_norm + cb_weight * cb_norm

    # Build results dataframe
    results = courses_df.set_index("course_id")[["name", "domain", "difficulty"]].copy()
    results["CF_score"]     = cf_norm.round(3)
    results["CB_score"]     = cb_norm.round(3)
    results["Hybrid_score"] = hybrid.round(3)
    results["Implicit_rating"] = [
        compute_implicit_rating(student, row)
        for _, row in courses_df.iterrows()
    ]

    return results.sort_values("Hybrid_score", ascending=False).head(top_n)


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def evaluate(n_splits=5):
    """
    K-Fold cross-validation: predict rating using CF, measure RMSE.
    Also compute Precision@K and Recall@K (threshold = 3.5).
    """
    rmse_list, precision_list, recall_list = [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(students_df):
        train_students = students_df.iloc[train_idx]["student_id"].tolist()
        test_students  = students_df.iloc[test_idx]["student_id"].tolist()

        train_matrix = rating_matrix.loc[train_students]
        course_sim   = pd.DataFrame(
            cosine_similarity(train_matrix.T),
            index=train_matrix.columns,
            columns=train_matrix.columns,
        )

        fold_rmse, fold_prec, fold_rec = [], [], []
        for sid in test_students:
            actual = rating_matrix.loc[sid]
            predicted_vals = []
            actual_vals    = []
            for cid in rating_matrix.columns:
                sims  = course_sim[cid].drop(cid)
                rated = train_matrix.mean()  # use training mean as proxy
                denom = sims.abs().sum()
                pred  = float((sims * rated).sum() / denom) if denom > 1e-6 else rated.mean()
                predicted_vals.append(np.clip(pred * 5, 0.5, 5.0))
                actual_vals.append(actual[cid])

            pred_arr   = np.array(predicted_vals)
            actual_arr = np.array(actual_vals)
            fold_rmse.append(np.sqrt(mean_squared_error(actual_arr, pred_arr)))

            # Precision & Recall @K=5
            k = 5
            threshold = 3.5
            top_k_pred = set(np.argsort(pred_arr)[-k:])
            relevant   = set(np.where(actual_arr >= threshold)[0])
            if top_k_pred:
                fold_prec.append(len(top_k_pred & relevant) / k)
            if relevant:
                fold_rec.append(len(top_k_pred & relevant) / len(relevant))

        rmse_list.append(np.mean(fold_rmse))
        precision_list.append(np.mean(fold_prec))
        recall_list.append(np.mean(fold_rec))

    return {
        "RMSE":       round(float(np.mean(rmse_list)), 4),
        "RMSE_std":   round(float(np.std(rmse_list)),  4),
        "Precision@5": round(float(np.mean(precision_list)), 4),
        "Recall@5":    round(float(np.mean(recall_list)),    4),
    }


# ─────────────────────────────────────────────
# 8. MAIN — TESTABLE CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  SMART COURSE RECOMMENDATION SYSTEM")
    print("  Waniya Syed (22k-4516) | Hafsa Atiqi (22k-4584)")
    print("=" * 65)

    # Show a few student profiles
    print("\n--- Sample Student Profiles ---")
    print(students_df.head(5).to_string(index=False))

    print("\n--- Course Catalogue ---")
    print(courses_df[["course_id", "name", "domain", "difficulty"]].to_string(index=False))

    print("\n--- Rating Matrix (first 5 students x 5 courses) ---")
    print(rating_matrix.iloc[:5, :5].to_string())

    # Demo recommendations for 3 students (first, middle, last)
    sample_ids = [
        students_df["student_id"].iloc[0],
        students_df["student_id"].iloc[len(students_df) // 2],
        students_df["student_id"].iloc[-1],
    ]
    for sid in sample_ids:
        student_row = students_df.set_index("student_id").loc[sid]
        print(f"\n{'='*65}")
        print(f"Recommendations for {sid} | GPA={student_row['gpa']} | "
              f"Math={student_row['math_score']} | Prog={student_row['prog_score']}")
        print(f"{'='*65}")
        recs = hybrid_recommend(sid, top_n=5)
        print(recs.to_string())

    # Evaluation
    print("\n--- Running 5-Fold Cross-Validation ---")
    metrics = evaluate(n_splits=5)
    print(f"  RMSE:       {metrics['RMSE']} (± {metrics['RMSE_std']})")
    print(f"  Precision@5: {metrics['Precision@5']}")
    print(f"  Recall@5:    {metrics['Recall@5']}")

    print("\nDone. Run `python recommendation_system.py` to reproduce results.")