import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import numpy as np
from typing import Dict, List, Tuple
import io

class CourseCoveringProblem:
    """Course covering optimization problem solver matching exact mathematical formulation."""
    
    def __init__(self, courses: List[str], professors: List[str], terms: List[str],
                 course_preferences: Dict[Tuple[str, str], float],  # c_ij (0-10 scale)
                 term_preferences: Dict[Tuple[str, str], float],    # t_jk (0-10 scale)
                 course_streams: Dict[Tuple[str, str], int],        # n_ik: streams per course per term
                 professor_total_load: Dict[str, int],              # b_j: total courses per professor
                 professor_term_limits: Dict[Tuple[str, str], int], # L_jk: max streams per prof per term (independent from b_j)
                 course_offerings: Dict[Tuple[str, str], int]):     # O_ik: 1 if course offered in term
        
        self.courses = courses  # S
        self.professors = professors  # P
        self.terms = terms  # T
        self.course_preferences = course_preferences  # c_ij (0-10, higher is better)
        self.term_preferences = term_preferences  # t_jk (0-10, higher is better)
        self.course_streams = course_streams  # n_ik
        self.professor_total_load = professor_total_load  # b_j (total teaching load)
        self.professor_term_limits = professor_term_limits  # L_jk (independent from b_j)
        self.course_offerings = course_offerings  # O_ik (parameter, not variable)
        
        self.model = None
        self.x_vars = {}  # x_ijk
        
        # Validate preference scores are in correct range
        self._validate_preferences()
        
    def _validate_preferences(self):
        """Validate that preference scores are in 0-10 range."""
        for (course, prof), score in self.course_preferences.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Course preference c_{{{course},{prof}}} = {score} must be in range [0,10]")
        
        for (prof, term), score in self.term_preferences.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Term preference t_{{{prof},{term}}} = {score} must be in range [0,10]")
    
    def build_model(self):
        """Build the optimization model matching the mathematical formulation exactly."""
        self.model = pulp.LpProblem("Course_Covering_Mathematical", pulp.LpMaximize)
        
        # Decision variables: x_ijk (only where course is offered)
        self.x_vars = {}
        for course in self.courses:
            for professor in self.professors:
                for term in self.terms:
                    # Only create variables where course is actually offered (O_ik = 1)
                    if self.course_offerings.get((course, term), 0) == 1:
                        self.x_vars[(course, professor, term)] = pulp.LpVariable(
                            f"x_{course}_{professor}_{term}", cat='Binary'
                        )
        
        # Objective function: Equation (1) - Maximize c_ij + t_jk preferences
        course_pref_term = pulp.lpSum([
            self.course_preferences.get((course, professor), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.terms
            if (course, professor, term) in self.x_vars  # Only for offered courses
        ])
        
        term_pref_term = pulp.lpSum([
            self.term_preferences.get((professor, term), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.terms
            if (course, professor, term) in self.x_vars  # Only for offered courses
        ])
        
        self.model += course_pref_term + term_pref_term
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints matching the mathematical formulation exactly."""
        
        # Constraint (2): Stream Load Constraint (Per Term) - L_jk limits
        # sum(n_ik * x_ijk) <= L_jk for all j, k
        for professor in self.professors:
            for term in self.terms:
                term_stream_load = pulp.lpSum([
                    self.course_streams.get((course, term), 0) * self.x_vars[(course, professor, term)]
                    for course in self.courses
                    if (course, professor, term) in self.x_vars  # Only for offered courses
                ])
                # L_jk is independent constraint - professor j's limit in term k
                max_streams_in_term = self.professor_term_limits.get((professor, term), 0)
                self.model += (
                    term_stream_load <= max_streams_in_term,
                    f"StreamLoad_L_{professor}_{term}"
                )
        
        # Constraint (3): Course Load Constraint (Total) - b_j limits  
        # sum(x_ijk) <= b_j for all j (total courses across all terms)
        for professor in self.professors:
            total_courses_assigned = pulp.lpSum([
                self.x_vars[(course, professor, term)]
                for course in self.courses
                for term in self.terms
                if (course, professor, term) in self.x_vars  # Only for offered courses
            ])
            # b_j is total teaching load for professor j (independent from L_jk)
            self.model += (
                total_courses_assigned <= self.professor_total_load[professor],
                f"TotalLoad_b_{professor}"
            )
        
        # Constraint (4): Course Offering Constraint - Each offered course gets exactly one academic
        # sum(x_ijk) = O_ik for all i, k where O_ik = 1
        for course in self.courses:
            for term in self.terms:
                if self.course_offerings.get((course, term), 0) == 1:  # Only if course is offered (O_ik = 1)
                    self.model += (
                        pulp.lpSum([
                            self.x_vars[(course, professor, term)]
                            for professor in self.professors
                            if (course, professor, term) in self.x_vars
                        ]) == 1,  # Exactly one professor must be assigned
                        f"CourseOffering_O_{course}_{term}"
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
                'professor_loads': {},
                'unassigned_offerings': [],
                'constraint_violations': self._analyze_infeasibility() if status == 'Infeasible' else None
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}  # {(course, term): professor}
        professor_loads = {
            prof: {
                'total_courses': 0,  # Tracks b_j constraint
                'streams_per_term': {term: 0 for term in self.terms},  # Tracks L_jk constraints
                'total_streams': 0
            } 
            for prof in self.professors
        }
        unassigned_offerings = []
        
        # Extract assignments
        for course in self.courses:
            for term in self.terms:
                if self.course_offerings.get((course, term), 0) == 1:  # Course should be offered
                    assigned = False
                    for professor in self.professors:
                        if (course, professor, term) in self.x_vars and self.x_vars[(course, professor, term)].varValue == 1:
                            assignments[(course, term)] = professor
                            professor_loads[professor]['total_courses'] += 1  # Count for b_j
                            streams_count = self.course_streams.get((course, term), 1)
                            professor_loads[professor]['streams_per_term'][term] += streams_count  # Count for L_jk
                            professor_loads[professor]['total_streams'] += streams_count
                            assigned = True
                            break
                    
                    if not assigned:
                        unassigned_offerings.append((course, term))
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'professor_loads': professor_loads,
            'unassigned_offerings': unassigned_offerings
        }
    
    def _analyze_infeasibility(self):
        """Analyze potential constraint violations if problem is infeasible."""
        violations = []
        
        # Check if total required streams exceed total available capacity
        total_required_streams = sum([
            self.course_streams.get((course, term), 1)
            for course in self.courses
            for term in self.terms
            if self.course_offerings.get((course, term), 0) == 1
        ])
        
        total_available_streams = sum([
            self.professor_term_limits.get((prof, term), 0)
            for prof in self.professors
            for term in self.terms
        ])
        
        if total_required_streams > total_available_streams:
            violations.append(f"Total required streams ({total_required_streams}) exceed total available capacity ({total_available_streams})")
        
        # Check total course constraints
        total_required_courses = sum([
            1 for course in self.courses
            for term in self.terms
            if self.course_offerings.get((course, term), 0) == 1
        ])
        
        total_available_courses = sum(self.professor_total_load.values())
        
        if total_required_courses > total_available_courses:
            violations.append(f"Total required courses ({total_required_courses}) exceed total available slots ({total_available_courses})")
        
        return violations


def main():
    st.set_page_config(
        page_title="RASTA-OP - Resource Assignment Scheduling Tool for Academic Optimization",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 RASTA-OP")
    st.markdown("**Resource Assignment Scheduling Tool for Academic Optimization**")
    st.markdown("Optimize faculty assignments with mathematical precision")
    st.markdown("---")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method:",
        ["Manual Input", "Excel Upload"]
    )
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'courses' not in st.session_state:
        st.session_state.courses = []
    if 'professors' not in st.session_state:
        st.session_state.professors = []
    if 'terms' not in st.session_state:
        st.session_state.terms = ['T1', 'T2', 'T3']
    
    # Navigation based on input method
    if input_method == "Excel Upload":
        if st.session_state.step == 1:
            show_excel_upload_step()
        elif st.session_state.step == 2:
            show_data_analysis_step()
        elif st.session_state.step == 3:
            show_results_step_fixed()  # Use fixed results function
    else:  # Manual Input
        if st.session_state.step == 1:
            show_setup_step()
        elif st.session_state.step == 2:
            show_constraints_step()
        elif st.session_state.step == 3:
            show_preferences_step()
        elif st.session_state.step == 4:
            show_results_step_fixed()  # Use fixed results function


def show_results_step_fixed():
    """Show results with proper matrix display - FIXED VERSION."""
    st.header("Optimization Results - Matrix Format")
    
    # Get data from session state
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    course_offerings = st.session_state.course_offerings
    course_streams = st.session_state.course_streams
    professor_total_load = st.session_state.professor_total_load
    professor_term_limits = st.session_state.professor_term_limits
    course_preferences = st.session_state.course_preferences
    term_preferences = st.session_state.term_preferences
    
    # Run optimization
    with st.spinner("Running optimization..."):
        try:
            problem = CourseCoveringProblem(
                courses=courses,
                professors=professors,
                terms=terms,
                course_preferences=course_preferences,
                term_preferences=term_preferences,
                course_streams=course_streams,
                professor_total_load=professor_total_load,
                professor_term_limits=professor_term_limits,
                course_offerings=course_offerings
            )
            
            solution = problem.solve()
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            return
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", solution['status'])
    with col2:
        if solution.get('objective_value'):
            st.metric("Objective Value", f"{solution['objective_value']:.1f}")
        else:
            st.metric("Objective Value", "N/A")
    with col3:
        unassigned_count = len(solution.get('unassigned_offerings', []))
        st.metric("Unassigned", unassigned_count)
    
    if solution['status'] == 'Optimal':
        st.success("Optimization successful!")
        
        # 1. MAIN MATRIX - ALL TERMS COMBINED
        st.subheader("📊 Assignment Matrix - All Terms Combined")
        st.markdown("**Courses (rows) × Staff (columns) | Shows: 1-TermCode or 0 for no assignment**")
        
        # Create matrix with proper structure
        matrix_all_terms = pd.DataFrame(
            index=courses,     # Course names as row index
            columns=professors, # Professor names as column headers
            dtype=object  # Changed to object to store strings like "1-T1"
        )
        # Initialize with zeros
        matrix_all_terms[:] = 0
        
        # Fill with assignments showing term information
        if solution.get('assignments'):
            for (course, term), professor in solution['assignments'].items():
                if course in matrix_all_terms.index and professor in matrix_all_terms.columns:
                    matrix_all_terms.loc[course, professor] = f"1-{term}"
        
        # Display matrix
        st.dataframe(
            matrix_all_terms,
            height=600,
            use_container_width=True
        )
        
        # Matrix statistics
        assignment_count = 0
        if solution.get('assignments'):
            assignment_count = len(solution['assignments'])
        
        courses_assigned = 0
        active_staff = 0
        
        if solution.get('assignments'):
            assigned_courses = set()
            active_professors = set()
            for (course, term), professor in solution['assignments'].items():
                assigned_courses.add(course)
                active_professors.add(professor)
            courses_assigned = len(assigned_courses)
            active_staff = len(active_professors)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assignments", assignment_count)
        with col2:
            st.metric("Courses Assigned", f"{courses_assigned}/{len(courses)}")
        with col3:
            st.metric("Active Staff", f"{active_staff}/{len(professors)}")
        
        st.markdown("---")
        
        # 2. TERM-SPECIFIC MATRICES
        st.subheader("📅 Term-Specific Assignment Matrices")
        
        # Create tabs for each term
        tabs = st.tabs([f"{term}" for term in terms])
        
        for i, term in enumerate(terms):
            with tabs[i]:
                st.markdown(f"**{term} Matrix: Courses × Staff**")
                
                # Create term matrix
                matrix_term = pd.DataFrame(
                    index=courses,
                    columns=professors,
                    dtype=int
                )
                matrix_term[:] = 0
                
                # Fill with term-specific assignments
                if solution.get('assignments'):
                    for (course, assignment_term), professor in solution['assignments'].items():
                        if assignment_term == term:
                            if course in matrix_term.index and professor in matrix_term.columns:
                                matrix_term.loc[course, professor] = 1
                
                # Display term matrix
                st.dataframe(
                    matrix_term,
                    height=500,
                    use_container_width=True
                )
                
                # Term statistics
                term_assignments = matrix_term.sum().sum()
                term_active_staff = (matrix_term.sum(axis=0) > 0).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{term} Assignments", int(term_assignments))
                with col2:
                    st.metric(f"{term} Active Staff", term_active_staff)
        
        # 3. Workload Summary
        st.subheader("👥 Staff Workload Summary")
        
        workload_list = []
        for prof in professors:
            prof_data = solution['professor_loads'][prof]
            total_courses = prof_data['total_courses']
            max_courses = professor_total_load[prof]
            
            # Count per term
            term_counts = {}
            if solution.get('assignments'):
                for term in terms:
                    count = sum(1 for (course, t), p in solution['assignments'].items() 
                              if p == prof and t == term)
                    term_counts[term] = count
            else:
                term_counts = {term: 0 for term in terms}
            
            workload_row = {
                'Staff': prof,
                'Total': f"{total_courses}/{max_courses}",
                'Util%': f"{(total_courses/max_courses)*100:.0f}%" if max_courses > 0 else "0%"
            }
            
            for term in terms:
                workload_row[f'{term}'] = term_counts[term]
            
            workload_list.append(workload_row)
        
        workload_df = pd.DataFrame(workload_list)
        st.dataframe(workload_df, hide_index=True, use_container_width=True)
        
    else:
        st.error(f"Optimization failed: {solution['status']}")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            if st.session_state.get('input_method') == "Excel Upload":
                st.session_state.step = 2
            else:
                st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# Add placeholder functions for other steps
def show_excel_upload_step():
    st.header("Excel Upload - Placeholder")
    st.write("Excel upload functionality would go here")
    if st.button("Next"):
        st.session_state.step = 2
        st.rerun()

def show_data_analysis_step():
    st.header("Data Analysis - Placeholder") 
    st.write("Data analysis would go here")
    if st.button("Run Optimization"):
        st.session_state.step = 3
        st.rerun()

def show_setup_step():
    st.header("Setup - Placeholder")
    st.write("Manual setup would go here")
    if st.button("Next"):
        st.session_state.step = 2
        st.rerun()

def show_constraints_step():
    st.header("Constraints - Placeholder")
    st.write("Constraints setup would go here")
    if st.button("Next"):
        st.session_state.step = 3
        st.rerun()

def show_preferences_step():
    st.header("Preferences - Placeholder")
    st.write("Preferences setup would go here")
    if st.button("Run Optimization"):
        st.session_state.step = 4
        st.rerun()


if __name__ == "__main__":
    main()
