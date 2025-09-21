import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import numpy as np
from typing import Dict, List, Tuple
import io

class CourseCoveringProblem:
    """Course covering optimization problem solver."""
    
    def __init__(self, courses: List[str], professors: List[str], 
                 preferences: Dict[Tuple[str, str], float],
                 max_load_per_professor: int = 2,
                 min_professors_per_course: int = 1):
        
        self.courses = courses
        self.professors = professors
        self.preferences = preferences
        self.max_load = max_load_per_professor
        self.min_coverage = min_professors_per_course
        
        # All professors are qualified for all courses
        self.qualified_faculty = {course: professors.copy() for course in courses}
        
        self.model = None
        self.x_vars = {}
        self.y_vars = {}
        
    def build_model(self):
        """Build the optimization model."""
        self.model = pulp.LpProblem("Course_Covering", pulp.LpMaximize)
        
        # Decision variables
        self.x_vars = {}
        for course in self.courses:
            for professor in self.professors:
                self.x_vars[(course, professor)] = pulp.LpVariable(
                    f"x_{course}_{professor}", cat='Binary'
                )
        
        self.y_vars = {}
        for course in self.courses:
            for k in range(1, len(self.professors) + 1):
                self.y_vars[(course, k)] = pulp.LpVariable(
                    f"y_{course}_{k}", cat='Binary'
                )
        
        # Objective function
        coverage_term = pulp.lpSum([
            k * self.y_vars[(course, k)]
            for course in self.courses
            for k in range(1, len(self.professors) + 1)
        ])
        
        preference_term = pulp.lpSum([
            self.preferences.get((course, professor), 0) * self.x_vars[(course, professor)]
            for course in self.courses
            for professor in self.professors
        ])
        
        self.model += 1000 * coverage_term + preference_term
        self._add_constraints()
        
    def _add_constraints(self):
        """Add constraints to the model."""
        # Coverage constraints
        for course in self.courses:
            assigned_professors = pulp.lpSum([
                self.x_vars[(course, prof)] for prof in self.professors
            ])
            
            required_coverage = pulp.lpSum([
                k * self.y_vars[(course, k)]
                for k in range(1, len(self.professors) + 1)
            ])
            
            self.model += assigned_professors >= required_coverage
            
            # Exactly one coverage level per course
            coverage_levels = pulp.lpSum([
                self.y_vars[(course, k)]
                for k in range(1, len(self.professors) + 1)
            ])
            self.model += coverage_levels == 1
        
        # Capacity constraints
        for professor in self.professors:
            assigned_courses = pulp.lpSum([
                self.x_vars[(course, professor)] for course in self.courses
            ])
            self.model += assigned_courses <= self.max_load
        
        # Minimum coverage constraint
        for course in self.courses:
            min_coverage = pulp.lpSum([
                k * self.y_vars[(course, k)]
                for k in range(self.min_coverage, len(self.professors) + 1)
            ])
            self.model += min_coverage >= self.min_coverage
    
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
                'coverage': {},
                'uncovered_courses': list(self.courses)
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}
        coverage = {}
        uncovered_courses = []
        
        for course in self.courses:
            assigned_profs = []
            for professor in self.professors:
                if self.x_vars[(course, professor)].varValue == 1:
                    assigned_profs.append(professor)
            if assigned_profs:
                assignments[course] = assigned_profs
        
        for course in self.courses:
            for k in range(1, len(self.professors) + 1):
                if self.y_vars[(course, k)].varValue == 1:
                    coverage[course] = k
                    break
        
        for course in self.courses:
            if course not in assignments or len(assignments[course]) == 0:
                uncovered_courses.append(course)
        
        professor_loads = {prof: 0 for prof in self.professors}
        for course, profs in assignments.items():
            for prof in profs:
                professor_loads[prof] += 1
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'coverage': coverage,
            'uncovered_courses': uncovered_courses,
            'professor_loads': professor_loads
        }


def main():
    st.set_page_config(
        page_title="Course Covering Optimizer",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Course Covering Optimizer")
    st.markdown("Optimize faculty assignments based on preferences and constraints")
    st.markdown("---")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'courses' not in st.session_state:
        st.session_state.courses = []
    if 'professors' not in st.session_state:
        st.session_state.professors = []
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {}
    
    # Navigation
    step = st.session_state.step
    
    # Step 1: Setup
    if step == 1:
        show_setup_step()
    
    # Step 2: Preferences
    elif step == 2:
        show_preferences_step()
    
    # Step 3: Results
    elif step == 3:
        show_results_step()


def show_setup_step():
    """Show the setup step."""
    st.header("Step 1: Setup Courses and Professors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courses")
        courses_input = st.text_area(
            "Enter courses (one per line):",
            value="Calculus I\nStatistics\nOperations Research\nSupply Chain\nFinance",
            height=150
        )
    
    with col2:
        st.subheader("Professors")
        professors_input = st.text_area(
            "Enter professors (one per line):",
            value="Dr. Smith\nDr. Johnson\nDr. Williams\nDr. Brown",
            height=150
        )
    
    # Parameters
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        max_load = st.number_input(
            "Maximum courses per professor:",
            min_value=1, max_value=10, value=2
        )
    
    with col2:
        min_coverage = st.number_input(
            "Minimum professors per course:",
            min_value=1, max_value=5, value=1
        )
    
    # Process input and move to next step
    if st.button("Next: Set Preferences", type="primary"):
        courses = [course.strip() for course in courses_input.split('\n') if course.strip()]
        professors = [prof.strip() for prof in professors_input.split('\n') if prof.strip()]
        
        if not courses or not professors:
            st.error("Please enter at least one course and one professor.")
        else:
            st.session_state.courses = courses
            st.session_state.professors = professors
            st.session_state.max_load = max_load
            st.session_state.min_coverage = min_coverage
            st.session_state.step = 2
            st.rerun()


def show_preferences_step():
    """Show the preferences step."""
    st.header("Step 2: Set Preference Scores")
    st.markdown("Set preference scores (1-5) for each professor-course pair:")
    st.markdown("**1 = Strongly Dislike, 3 = Neutral, 5 = Strongly Prefer**")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    
    # Create preference matrix
    preferences_data = []
    
    # Use columns for better layout
    n_cols = min(3, len(professors))  # Max 3 columns for readability
    cols = st.columns(n_cols)
    
    col_idx = 0
    for professor in professors:
        with cols[col_idx % n_cols]:
            st.subheader(f"{professor}")
            
            for course in courses:
                # Get existing preference or default to 3
                existing_pref = st.session_state.preferences.get((course, professor), 3)
                
                pref = st.slider(
                    f"{course}",
                    min_value=1, max_value=5, value=existing_pref,
                    key=f"pref_{course}_{professor}"
                )
                
                st.session_state.preferences[(course, professor)] = pref
                preferences_data.append({
                    'Course': course,
                    'Professor': professor,
                    'Preference': pref
                })
        
        col_idx += 1
    
    # Show preference matrix
    st.subheader("Preference Matrix Overview")
    pref_df = pd.DataFrame(preferences_data)
    pivot_df = pref_df.pivot(index='Course', columns='Professor', values='Preference')
    
    # Color code the matrix
    fig = px.imshow(
        pivot_df.values,
        labels=dict(x="Professor", y="Course", color="Preference"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="RdYlGn",
        range_color=[1, 5],
        title="Preference Heatmap (Green = High Preference, Red = Low Preference)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
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
    preferences = st.session_state.preferences
    max_load = st.session_state.max_load
    min_coverage = st.session_state.min_coverage
    
    # Run optimization
    with st.spinner("Running optimization..."):
        problem = CourseCoveringProblem(
            courses=courses,
            professors=professors,
            preferences=preferences,
            max_load_per_professor=max_load,
            min_professors_per_course=min_coverage
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
        
        # Course assignments
        st.subheader("Course Assignments")
        assignments_data = []
        for course in courses:
            assigned_profs = solution['assignments'].get(course, [])
            if assigned_profs:
                assignments_data.append({
                    'Course': course,
                    'Assigned Professors': ', '.join(assigned_profs),
                    'Coverage Level': len(assigned_profs),
                    'Status': '✅ Covered'
                })
            else:
                assignments_data.append({
                    'Course': course,
                    'Assigned Professors': 'None',
                    'Coverage Level': 0,
                    'Status': '❌ Uncovered'
                })
        
        assignments_df = pd.DataFrame(assignments_data)
        st.dataframe(assignments_df, use_container_width=True, hide_index=True)
        
        # Professor workloads
        st.subheader("Professor Workload Distribution")
        workload_data = []
        for prof in professors:
            load = solution['professor_loads'].get(prof, 0)
            assigned_courses = [
                course for course, profs in solution['assignments'].items() 
                if prof in profs
            ]
            workload_data.append({
                'Professor': prof,
                'Courses Assigned': load,
                'Course Names': ', '.join(assigned_courses) if assigned_courses else 'None',
                'Utilization %': (load / max_load) * 100
            })
        
        workload_df = pd.DataFrame(workload_data)
        
        # Workload chart
        fig = px.bar(
            workload_df, 
            x='Professor', 
            y='Courses Assigned',
            title='Teaching Load per Professor',
            color='Utilization %',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Course Names']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Professor details table
        st.subheader("Detailed Professor Assignments")
        st.dataframe(workload_df, use_container_width=True, hide_index=True)
        
        # Coverage analysis
        st.subheader("Coverage Analysis")
        coverage_data = {
            'Covered': len([c for c in courses if c in solution['assignments']]),
            'Uncovered': len(solution.get('uncovered_courses', []))
        }
        
        if coverage_data['Uncovered'] > 0:
            fig = px.pie(
                values=list(coverage_data.values()),
                names=list(coverage_data.keys()),
                title='Course Coverage Status',
                color_discrete_map={'Covered': 'green', 'Uncovered': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Uncovered Courses")
            for course in solution.get('uncovered_courses', []):
                st.error(f"**{course}** - No assignment possible with current constraints")
        else:
            st.success("All courses are successfully covered!")
        
        # Download results
        st.subheader("Download Results")
        csv_buffer = io.StringIO()
        assignments_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Assignment Results (CSV)",
            data=csv_data,
            file_name="course_assignments.csv",
            mime="text/csv"
        )
        
    else:
        st.error(f"Optimization failed: {solution['status']}")
        st.write("Possible reasons:")
        st.write("- Constraints are too restrictive")
        st.write("- Not enough professors for the required coverage")
        st.write("- Conflicting requirements")
    
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
