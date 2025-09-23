import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import numpy as np
from typing import Dict, List, Tuple
import io

class CourseCoveringProblem:
    """Course covering optimization problem solver with terms and class-based constraints."""
    
    def __init__(self, courses: List[str], professors: List[str], terms: List[str],
                 course_preferences: Dict[Tuple[str, str], float],
                 term_preferences: Dict[Tuple[str, str], float],
                 course_classes: Dict[str, int],
                 professor_max_courses: Dict[str, int],  # b_j: individual limits per professor
                 course_allowed_terms: Dict[str, List[str]],  # Which terms each course can be offered
                 max_classes_per_term: int = 3):
        
        self.courses = courses
        self.professors = professors
        self.terms = terms
        self.course_preferences = course_preferences
        self.term_preferences = term_preferences
        self.course_classes = course_classes  # n_i: classes per course
        self.professor_max_courses = professor_max_courses  # b_j: individual limits per professor
        self.course_allowed_terms = course_allowed_terms  # Which terms each course can be offered
        self.max_classes_term = max_classes_per_term  # L = 3
        
        self.model = None
        self.x_vars = {}  # x_ijk
        self.o_vars = {}  # O_ik
        
    def build_model(self):
        """Build the optimization model."""
        self.model = pulp.LpProblem("Course_Covering_Multi_Term", pulp.LpMaximize)
        
        # Decision variables
        # x_ijk: 1 if professor j teaches course i in term k (only for allowed terms)
        self.x_vars = {}
        for course in self.courses:
            for professor in self.professors:
                for term in self.course_allowed_terms[course]:  # Only allowed terms
                    self.x_vars[(course, professor, term)] = pulp.LpVariable(
                        f"x_{course}_{professor}_{term}", cat='Binary'
                    )
        
        # O_ik: 1 if course i is offered in term k (only for allowed terms)
        self.o_vars = {}
        for course in self.courses:
            for term in self.course_allowed_terms[course]:  # Only allowed terms
                self.o_vars[(course, term)] = pulp.LpVariable(
                    f"O_{course}_{term}", cat='Binary'
                )
        
        # Objective function: maximize course preferences + term preferences (only for allowed terms)
        course_pref_term = pulp.lpSum([
            self.course_preferences.get((course, professor), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.course_allowed_terms[course]  # Only allowed terms
        ])
        
        term_pref_term = pulp.lpSum([
            self.term_preferences.get((professor, term), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.course_allowed_terms[course]  # Only allowed terms
        ])
        
        self.model += course_pref_term + term_pref_term
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints to the model."""
        
        # 1. One professor per course per term (only for allowed terms)
        for course in self.courses:
            for term in self.course_allowed_terms[course]:  # Only allowed terms
                self.model += (
                    pulp.lpSum([self.x_vars[(course, professor, term)] for professor in self.professors]) <= 1,
                    f"SingleProf_{course}_{term}"
                )
        
        # 2. Faculty teaching load constraint (per term) - class-based
        for professor in self.professors:
            for term in self.terms:  # Check all terms for professor workload
                term_load = pulp.lpSum([
                    self.course_classes[course] * self.x_vars[(course, professor, term)]
                    for course in self.courses
                    if term in self.course_allowed_terms[course]  # Only if course allowed in this term
                ])
                self.model += (
                    term_load <= self.max_classes_term,
                    f"ClassLoad_{professor}_{term}"
                )
        
        # 3. Faculty teaching load constraint (total courses) - individual limits
        for professor in self.professors:
            total_courses = pulp.lpSum([
                self.x_vars[(course, professor, term)]
                for course in self.courses
                for term in self.course_allowed_terms[course]  # Only allowed terms
            ])
            self.model += (
                total_courses <= self.professor_max_courses[professor],  # Use individual limit b_j
                f"TotalCourses_{professor}"
            )
        
        # 4. Course offering constraint: O_ik = sum of x_ijk (only for allowed terms)
        for course in self.courses:
            for term in self.course_allowed_terms[course]:  # Only allowed terms
                self.model += (
                    pulp.lpSum([self.x_vars[(course, professor, term)] for professor in self.professors]) == self.o_vars[(course, term)],
                    f"Offering_{course}_{term}"
                )
    
    def solve(self):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.model.solve(solver)
        
        status = pulp.LpStatus[self.model.status]
        
        if status == 'Optimal':
            return self._extract_solution()
        else:
            return {
                'status': status,
                'objective_value': None,
                'assignments': {},
                'course_offerings': {},
                'uncovered_courses': list(self.courses),
                'professor_loads': {}
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}  # {(course, term): professor}
        course_offerings = {}  # {course: [terms]}
        professor_loads = {prof: {'courses': 0, 'classes_per_term': {term: 0 for term in self.terms}} for prof in self.professors}
        
        # Extract assignments (only for allowed terms)
        for course in self.courses:
            for term in self.course_allowed_terms[course]:  # Only allowed terms
                for professor in self.professors:
                    if self.x_vars[(course, professor, term)].varValue == 1:
                        assignments[(course, term)] = professor
                        professor_loads[professor]['courses'] += 1
                        professor_loads[professor]['classes_per_term'][term] += self.course_classes[course]
        
        # Extract course offerings (only for allowed terms)
        for course in self.courses:
            offered_terms = []
            for term in self.course_allowed_terms[course]:  # Only allowed terms
                if self.o_vars[(course, term)].varValue == 1:
                    offered_terms.append(term)
            if offered_terms:
                course_offerings[course] = offered_terms
        
        # Find uncovered courses (courses not offered in any allowed term)
        uncovered_courses = [course for course in self.courses if course not in course_offerings]
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'course_offerings': course_offerings,
            'uncovered_courses': uncovered_courses,
            'professor_loads': professor_loads
        }


def main():
    st.set_page_config(
        page_title="Course Covering Optimizer - Multi Term",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Course Covering Optimizer - Multi Term")
    st.markdown("Optimize faculty assignments across terms based on course and term preferences")
    st.markdown("---")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'courses' not in st.session_state:
        st.session_state.courses = []
    if 'professors' not in st.session_state:
        st.session_state.professors = []
    if 'terms' not in st.session_state:
        st.session_state.terms = []
    if 'course_preferences' not in st.session_state:
        st.session_state.course_preferences = {}
    if 'term_preferences' not in st.session_state:
        st.session_state.term_preferences = {}
    if 'course_classes' not in st.session_state:
        st.session_state.course_classes = {}
    if 'professor_max_courses' not in st.session_state:
        st.session_state.professor_max_courses = {}
    if 'course_allowed_terms' not in st.session_state:
        st.session_state.course_allowed_terms = {}
    
    # Navigation
    step = st.session_state.step
    
    if step == 1:
        show_setup_step()
    elif step == 2:
        show_preferences_step()
    elif step == 3:
        show_results_step()


def show_setup_step():
    """Show the setup step."""
    st.header("Step 1: Setup Courses, Professors, and Terms")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Courses")
        courses_input = st.text_area(
            "Enter courses (one per line):",
            value="ACTL1\nACTL2\nACTL3\nACTL4\nACTL5",
            height=120
        )
    
    with col2:
        st.subheader("Professors")
        professors_input = st.text_area(
            "Enter professors (one per line):",
            value="Jonathan\nJK\nPatrick\nAndres",
            height=120
        )
    
    with col3:
        st.subheader("Terms")
        terms_input = st.text_area(
            "Enter terms (one per line):",
            value="T1\nT2\nT3",
            height=120
        )
    
    # Course classes configuration
    st.subheader("Course Configuration")
    courses_preview = [course.strip() for course in courses_input.split('\n') if course.strip()]
    terms_preview = [term.strip() for term in terms_input.split('\n') if term.strip()]
    
    if courses_preview and terms_preview:
        st.write("Configure each course:")
        course_classes = {}
        course_allowed_terms = {}
        
        cols = st.columns(min(2, len(courses_preview)))
        for idx, course in enumerate(courses_preview):
            with cols[idx % len(cols)]:
                st.write(f"**{course}**")
                
                # Number of classes for this course
                classes = st.number_input(
                    f"Classes:",
                    min_value=1, max_value=5, value=2,
                    key=f"classes_{course}"
                )
                course_classes[course] = classes
                
                # Which terms this course can be offered
                allowed_terms = st.multiselect(
                    f"Allowed terms:",
                    options=terms_preview,
                    default=[terms_preview[0]],  # Default to first term
                    key=f"allowed_terms_{course}"
                )
                course_allowed_terms[course] = allowed_terms
                
                if not allowed_terms:
                    st.warning(f"Please select at least one term for {course}")
        
        # Show course-term matrix
        if course_classes and course_allowed_terms:
            st.write("**Course-Term Availability Matrix:**")
            matrix_data = []
            for course in courses_preview:
                row_data = {'Course': course, 'Classes': course_classes[course]}
                for term in terms_preview:
                    row_data[term] = '✓' if term in course_allowed_terms.get(course, []) else '✗'
                matrix_data.append(row_data)
            
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    
    # Professor loading configuration
    st.subheader("Professor Loading Configuration")
    professors_preview = [prof.strip() for prof in professors_input.split('\n') if prof.strip()]
    
    if professors_preview:
        st.write("Set maximum courses for each professor:")
        professor_max_courses = {}
        
        cols = st.columns(min(4, len(professors_preview)))
        for idx, professor in enumerate(professors_preview):
            with cols[idx % len(cols)]:
                max_courses_prof = st.number_input(
                    f"Max courses for {professor}:",
                    min_value=1, max_value=8, value=4,
                    key=f"max_courses_{professor}"
                )
                professor_max_courses[professor] = max_courses_prof
    
    # Parameters
    st.subheader("Global Parameters")
    max_classes = st.number_input(
        "Maximum classes per professor per term:",
        min_value=1, max_value=6, value=3
    )
    
    # Process input and move to next step
    if st.button("Next: Set Preferences", type="primary"):
        courses = [course.strip() for course in courses_input.split('\n') if course.strip()]
        professors = [prof.strip() for prof in professors_input.split('\n') if prof.strip()]
        terms = [term.strip() for term in terms_input.split('\n') if term.strip()]
        
        if not courses or not professors or not terms:
            st.error("Please enter at least one course, professor, and term.")
        elif not all(course_allowed_terms.values()):
            st.error("Please select allowed terms for all courses.")
        else:
            st.session_state.courses = courses
            st.session_state.professors = professors
            st.session_state.terms = terms
            st.session_state.course_classes = course_classes
            st.session_state.course_allowed_terms = course_allowed_terms
            st.session_state.professor_max_courses = professor_max_courses
            st.session_state.max_classes = max_classes
            st.session_state.step = 2
            st.rerun()


def show_preferences_step():
    """Show the preferences step."""
    st.header("Step 2: Set Preference Scores")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    
    # Course Preferences
    st.subheader("Course Preferences")
    st.markdown("Set how much each professor prefers to teach each course:")
    st.markdown("**0 = Cannot Teach, 1 = Can Teach (Low Preference), 3 = Strongly Prefer**")
    
    # Create course preference matrix
    course_pref_data = []
    
    for professor in professors:
        st.write(f"**{professor}'s Course Preferences:**")
        cols = st.columns(min(4, len(courses)))
        
        for idx, course in enumerate(courses):
            with cols[idx % len(cols)]:
                existing_pref = st.session_state.course_preferences.get((course, professor), 1)
                pref = st.selectbox(
                    f"{course}",
                    options=[0, 1, 3],
                    index=[0, 1, 3].index(existing_pref) if existing_pref in [0, 1, 3] else 1,
                    key=f"course_pref_{course}_{professor}"
                )
                st.session_state.course_preferences[(course, professor)] = pref
                course_pref_data.append({
                    'Course': course,
                    'Professor': professor,
                    'Preference': pref
                })
    
    # Term Preferences
    st.subheader("Term Preferences")
    st.markdown("Set how much each professor prefers to teach in each term:")
    
    for professor in professors:
        st.write(f"**{professor}'s Term Preferences:**")
        cols = st.columns(len(terms))
        
        for idx, term in enumerate(terms):
            with cols[idx]:
                existing_pref = st.session_state.term_preferences.get((professor, term), 1)
                pref = st.selectbox(
                    f"{term}",
                    options=[0, 1, 3],
                    index=[0, 1, 3].index(existing_pref) if existing_pref in [0, 1, 3] else 1,
                    key=f"term_pref_{professor}_{term}"
                )
                st.session_state.term_preferences[(professor, term)] = pref
    
    # Show preference matrices
    st.subheader("Preference Overview")
    
    # Course preference heatmap
    if course_pref_data:
        course_pref_df = pd.DataFrame(course_pref_data)
        course_pivot = course_pref_df.pivot(index='Course', columns='Professor', values='Preference')
        
        fig1 = px.imshow(
            course_pivot.values,
            labels=dict(x="Professor", y="Course", color="Course Preference"),
            x=course_pivot.columns,
            y=course_pivot.index,
            color_continuous_scale="RdYlGn",
            range_color=[0, 3],
            title="Course Preference Matrix"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Setup"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Run Optimization →", type="primary"):
            st.session_state.step = 3
            st.rerun()


def show_results_step():
    """Show the results step."""
    st.header("Step 3: Optimization Results")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    course_preferences = st.session_state.course_preferences
    term_preferences = st.session_state.term_preferences
    course_classes = st.session_state.course_classes
    course_allowed_terms = st.session_state.course_allowed_terms
    professor_max_courses = st.session_state.professor_max_courses
    max_classes = st.session_state.max_classes
    
    # Run optimization
    with st.spinner("Running optimization..."):
        problem = CourseCoveringProblem(
            courses=courses,
            professors=professors,
            terms=terms,
            course_preferences=course_preferences,
            term_preferences=term_preferences,
            course_classes=course_classes,
            professor_max_courses=professor_max_courses,  # Individual limits per professor
            course_allowed_terms=course_allowed_terms,  # Specific allowed terms per course
            max_classes_per_term=max_classes
        )
        
        solution = problem.solve()
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", solution['status'])
    with col2:
        st.metric("Objective Value", f"{solution.get('objective_value', 0):.1f}")
    with col3:
        uncovered_count = len(solution.get('uncovered_courses', []))
        st.metric("Uncovered Courses", uncovered_count)
    
    if solution['status'] == 'Optimal':
        
        # Course assignments by term
        st.subheader("Course Assignments by Term")
        
        assignments_data = []
        for term in terms:
            term_assignments = []
            for course in courses:
                if (course, term) in solution['assignments']:
                    professor = solution['assignments'][(course, term)]
                    classes = course_classes[course]
                    term_assignments.append(f"{course} → {professor} ({classes} classes)")
                    assignments_data.append({
                        'Term': term,
                        'Course': course,
                        'Professor': professor,
                        'Classes': classes,
                        'Status': 'Assigned'
                    })
            
            if term_assignments:
                st.write(f"**{term}:**")
                for assignment in term_assignments:
                    st.write(f"  • {assignment}")
            else:
                st.write(f"**{term}:** No courses assigned")
        
        # Assignments dataframe
        if assignments_data:
            assignments_df = pd.DataFrame(assignments_data)
            st.dataframe(assignments_df, use_container_width=True, hide_index=True)
        
        # Professor workload analysis
        st.subheader("Professor Workload Analysis")
        
        workload_summary_data = []
        workload_data = []
        for professor in professors:
            prof_data = solution['professor_loads'][professor]
            total_courses = prof_data['courses']
            max_courses_allowed = professor_max_courses[professor]  # Individual limit
            
            workload_summary_data.append({
                'Professor': professor,
                'Total Courses': total_courses,
                'Max Allowed': max_courses_allowed,
                'Course Utilization %': (total_courses / max_courses_allowed) * 100
            })
            
            for term in terms:
                classes_in_term = prof_data['classes_per_term'][term]
                workload_data.append({
                    'Professor': professor,
                    'Term': term,
                    'Classes': classes_in_term,
                    'Class Utilization %': (classes_in_term / max_classes) * 100
                })
        
        # Show professor workload summary
        workload_summary_df = pd.DataFrame(workload_summary_data)
        st.write("**Professor Course Loads:**")
        st.dataframe(workload_summary_df, use_container_width=True, hide_index=True)
        
        workload_df = pd.DataFrame(workload_data)
        
        # Workload heatmap
        workload_pivot = workload_df.pivot(index='Professor', columns='Term', values='Classes')
        fig2 = px.imshow(
            workload_pivot.values,
            labels=dict(x="Term", y="Professor", color="Classes"),
            x=workload_pivot.columns,
            y=workload_pivot.index,
            color_continuous_scale="Blues",
            title="Professor Workload by Term (Classes)"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Course offerings summary
        st.subheader("Course Offerings Summary")
        
        if solution['course_offerings']:
            offering_data = []
            for course, offered_terms in solution['course_offerings'].items():
                offering_data.append({
                    'Course': course,
                    'Offered Terms': ', '.join(offered_terms),
                    'Number of Terms': len(offered_terms),
                    'Total Classes': course_classes[course]
                })
            
            offerings_df = pd.DataFrame(offering_data)
            st.dataframe(offerings_df, use_container_width=True, hide_index=True)
        
        # Uncovered courses
        if solution.get('uncovered_courses'):
            st.subheader("⚠️ Uncovered Courses")
            for course in solution['uncovered_courses']:
                st.error(f"**{course}** ({course_classes[course]} classes) - No assignment possible")
        else:
            st.success("🎉 All courses are covered!")
        
        # Download results
        st.subheader("Download Results")
        if assignments_data:
            csv_buffer = io.StringIO()
            assignments_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Assignment Results (CSV)",
                data=csv_data,
                file_name="multi_term_course_assignments.csv",
                mime="text/csv"
            )
        
    else:
        st.error(f"Optimization failed: {solution['status']}")
        st.write("Possible reasons:")
        st.write("- Class load constraints are too restrictive (try increasing max classes per term)")
        st.write("- Not enough professors for the required course coverage")
        st.write("- Preference scores are too low (all 0s)")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Preferences"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("🔄 Start Over"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
